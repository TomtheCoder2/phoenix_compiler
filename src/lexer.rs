// src/lexer.rs

use crate::token::Token;

pub struct Lexer<'a> {
    input: std::str::Chars<'a>,
    current_char: Option<char>,
    // Add peek functionality - useful for identifiers vs keywords
    peek_char: Option<char>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input: input.chars(),
            current_char: None,
            peek_char: None, // Initialize peek
        };
        lexer.advance(); // Load current_char
        lexer.advance(); // Load peek_char
        lexer
    }

    // Advance now updates both current and peek
    fn advance(&mut self) {
        self.current_char = self.peek_char;
        self.peek_char = self.input.next();
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let token = match self.current_char {
            // Single char tokens
            Some('+') => Token::Plus,
            Some('-') => Token::Minus,
            Some('*') => Token::Star,
            Some('/') => {
                // Check for comment first
                if self.peek_char == Some('/') {
                    self.advance(); // Consume first /
                    self.advance(); // Consume second /
                                    // Consume until newline
                    while let Some(ch) = self.current_char {
                        if ch == '\n' {
                            break;
                        }
                        self.advance();
                    }
                    // After loop, current_char is potentially '\n' or None.
                    // We need the token *after* the comment.
                    return self.next_token(); // Recurse
                } else {
                    Token::Slash // Regular division
                }
            }
            Some('(') => Token::LParen,
            Some(')') => Token::RParen,
            Some(';') => Token::Semicolon,
            Some(',') => Token::Comma,
            Some('{') => Token::LBrace,
            Some('}') => Token::RBrace,
            Some(':') => Token::Colon,

            // Two-char tokens (check peek_char)
            Some('=') => {
                if self.peek_char == Some('=') {
                    self.advance(); // Consume second '='
                    Token::Equal // '=='
                } else {
                    Token::Assign // '='
                }
            }
            Some('!') => {
                if self.peek_char == Some('=') {
                    self.advance(); // Consume '='
                    Token::NotEqual // '!='
                } else {
                    Token::Bang // Handle '!' later if needed for logical NOT
                    // Token::Illegal('!') // For now, '!' alone is illegal
                }
            }
            Some('<') => {
                if self.peek_char == Some('=') {
                    self.advance(); // Consume '='
                    Token::LessEqual // '<='
                } else {
                    Token::LessThan // '<'
                }
            }
            Some('>') => {
                if self.peek_char == Some('=') {
                    self.advance(); // Consume '='
                    Token::GreaterEqual // '>='
                } else {
                    Token::GreaterThan // '>'
                }
            }

            // Identifiers / Keywords / Bool Literals
            Some(ch) if is_identifier_start(ch) => {
                let ident = self.read_identifier();
                // Check keywords BEFORE returning identifier
                return match ident.as_str() {
                    "let" => Token::Let,
                    "var" => Token::Var,
                    "fun" => Token::Fun,
                    "true" => Token::BoolLiteral(true),
                    "false" => Token::BoolLiteral(false),
                    "if" => Token::If,
                    "else" => Token::Else,
                    "while" => Token::While,
                    // Check type names? Optional.
                    // type_name if keyword_to_type(type_name).is_some() => {
                    //     Token::Type(keyword_to_type(type_name).unwrap()) // Or specific TypeInt etc.
                    // }
                    _ => Token::Identifier(ident),
                };
            }

            // Numbers (Integer or Float)
            Some(ch) if ch.is_digit(10) => {
                // If it starts with a digit, it could be Int or Float
                return self.read_number(); // read_number determines Int or Float
            }
            Some('.') => {
                // Handle floats starting with '.'
                if self.peek_char.map_or(false, |pc| pc.is_digit(10)) {
                    return self.read_number(); // Let read_number handle ".5"
                } else {
                    Token::Illegal('.') // '.' alone is illegal
                }
            }

            Some(ch) => Token::Illegal(ch),
            None => Token::Eof,
        };

        self.advance(); // Move ahead for single-or-double-char tokens handled above
        token
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    // Reads a sequence of letters/digits/_ starting from current_char
    fn read_identifier(&mut self) -> String {
        let mut identifier = String::new();
        while let Some(ch) = self.current_char {
            if is_identifier_continue(ch) {
                identifier.push(ch);
                self.advance(); // Consume identifier char
            } else {
                break; // Stop if we hit a non-identifier character
            }
        }
        // We return the identifier string. The advance() call inside the loop
        // means current_char is already positioned *after* the identifier.
        identifier
    }

    fn read_number(&mut self) -> Token {
        let mut number_str = String::new();
        let mut is_float = false;

        while let Some(ch) = self.current_char {
            if ch.is_digit(10) {
                number_str.push(ch);
            } else if ch == '.' {
                // Check if we already saw a '.' or if the next char isn't a digit (e.g. "1..", "1.a")
                if is_float || !self.peek_char.map_or(false, |pc| pc.is_digit(10)) {
                    // If already float or next is not digit, the number ends here.
                    // The '.' might be illegal or start another token.
                    // Let parser handle "1..". We just return the number found so far.
                    break; // Stop reading the number
                }
                // If peek is digit, it's a float, add the '.'
                number_str.push(ch);
                is_float = true;
            } else {
                // Not a digit or a valid '.'
                break;
            }
            self.advance(); // Consume digit or '.'
        }

        // Now, current_char is the char *after* the number literal

        if is_float {
            match number_str.parse::<f64>() {
                Ok(num) => Token::FloatNum(num),
                // Should be rare if logic above is correct
                Err(_) => Token::Illegal(number_str.chars().next().unwrap_or('?')),
            }
        } else {
            // Didn't find '.', parse as i64
            match number_str.parse::<i64>() {
                Ok(num) => Token::IntNum(num),
                // Could fail if number is too large for i64
                Err(_) => {
                    // Maybe try parsing as f64 if i64 fails? Or report overflow?
                    // For now, treat as illegal if i64 parse fails.
                    // Could also introduce BigInt type later.
                    Token::Illegal(number_str.chars().next().unwrap_or('?'))
                }
            }
        }
    }
}

// Helper functions for identifier characters
fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_identifier_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

// --- Add tests for new tokens ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Token;

    #[test]
    fn test_let_sequence() {
        let input = "let x = 10; let y = x + 5;";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::IntNum(10),
            Token::Semicolon,
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Assign,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::IntNum(5),
            Token::Semicolon,
            Token::Eof,
        ];

        for expected_token in tokens {
            let actual_token = lexer.next_token();
            assert_eq!(actual_token, expected_token);
        }
    }

    #[test]
    fn test_float_parsing_dot_five() {
        let input = ".5 + 1";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::FloatNum(0.5),
            Token::Plus,
            Token::IntNum(1),
            Token::Eof,
        ];
        for expected_token in tokens {
            let actual_token = lexer.next_token();
            assert_eq!(actual_token, expected_token);
        }
    }

    #[test]
    // fn test_float_parsing_trailing_dot_error() {
    //     // Assuming "5." is not a valid number on its own for now
    //     // (Some lexers might allow it, parsing would handle it)
    //     let input = "5.";
    //     let mut lexer = Lexer::new(input);
    //     // Our current parse will likely succeed `5.`.parse::<f64>() ok -> 5.0
    //     // If we wanted `5.` to be illegal, read_number needs adjustment
    //     // Let's keep it simple and assume `5.` parses as 5.0 for now.
    //     assert_eq!(lexer.next_token(), Token::IntNum(5));
    //     assert_eq!(lexer.next_token(), Token::Eof);
    //
    //     // Test case for an clearly illegal sequence starting with '.'
    //     let input = " . ";
    //     let mut lexer = Lexer::new(input);
    //     assert_eq!(lexer.next_token(), Token::Illegal('.'));
    // }
    #[test]
    fn test_function_tokens() {
        let input = "fun add(a, b) { a + b; }";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::Fun,
            Token::Identifier("add".to_string()),
            Token::LParen,
            Token::Identifier("a".to_string()),
            Token::Comma,
            Token::Identifier("b".to_string()),
            Token::RParen,
            Token::LBrace,
            Token::Identifier("a".to_string()),
            Token::Plus,
            Token::Identifier("b".to_string()),
            Token::Semicolon,
            Token::RBrace,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }

    #[test]
    fn test_comments() {
        let input = "let x = 10; // This is a comment\nlet y = x + 5;";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::IntNum(10),
            Token::Semicolon,
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Assign,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::IntNum(5),
            Token::Semicolon,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }

    #[test]
    fn test_int_float_bool_tokens() {
        let input = "123 45.6 true false 999";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::IntNum(123),
            Token::FloatNum(45.6),
            Token::BoolLiteral(true),
            Token::BoolLiteral(false),
            Token::IntNum(999),
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }

    #[test]
    fn test_comparison_tokens() {
        let input = "= == != < <= > >=";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::Assign,
            Token::Equal,
            Token::NotEqual,
            Token::LessThan,
            Token::LessEqual,
            Token::GreaterThan,
            Token::GreaterEqual,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }

    #[test]
    fn test_type_annotation_token() {
        let input = "let x: int = 10;";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()), // Treat type name as identifier for now
            Token::Assign,
            Token::IntNum(10),
            Token::Semicolon,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }

    #[test]
    fn test_var_keyword() {
        let input = "var counter = 0;";
        let mut lexer = Lexer::new(input);
        let tokens = vec![
            Token::Var,
            Token::Identifier("counter".to_string()),
            Token::Assign,
            Token::IntNum(0),
            Token::Semicolon,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }

    #[test]
    fn test_if_else_tokens() {
        let input = "if (x < 10) { x = x + 1; } else { x = 0; }";
        let mut l = Lexer::new(input);
        let tokens = vec![
            Token::If,
            Token::LParen,
            Token::Identifier("x".into()),
            Token::LessThan,
            Token::IntNum(10),
            Token::RParen,
            Token::LBrace,
            Token::Identifier("x".into()),
            Token::Assign,
            Token::Identifier("x".into()),
            Token::Plus,
            Token::IntNum(1),
            Token::Semicolon,
            Token::RBrace,
            Token::Else,
            Token::LBrace,
            Token::Identifier("x".into()),
            Token::Assign,
            Token::IntNum(0),
            Token::Semicolon,
            Token::RBrace,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token(), expected);
        }
    }

    #[test]
    fn test_bang_operator() {
        let input = "!true != false";
        let mut l = Lexer::new(input);
        let tokens = vec![
            Token::Bang, Token::BoolLiteral(true), Token::NotEqual, Token::BoolLiteral(false), Token::Eof,
        ];
        for expected in tokens { assert_eq!(l.next_token(), expected); }
    }

    #[test]
    fn test_while_keyword() {
        let input = "while (i < 10) { i = i + 1; }";
        let mut l = Lexer::new(input);
        let tokens = vec![
            Token::While, Token::LParen, Token::Identifier("i".into()), Token::LessThan, Token::IntNum(10), Token::RParen,
            Token::LBrace, Token::Identifier("i".into()), Token::Assign, Token::Identifier("i".into()), Token::Plus, Token::IntNum(1), Token::Semicolon, Token::RBrace,
            Token::Eof,
        ];
        for expected in tokens { assert_eq!(l.next_token(), expected); }
    }
}

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
            Some('=') => Token::Assign,
            Some('+') => Token::Plus,
            Some('-') => Token::Minus,
            Some('*') => Token::Star,
            Some('/') if self.peek_char == Some('/') => {
                // Handle comments
                while let Some(next_ch) = self.peek_char {
                    if next_ch == '\n' {
                        break; // End of comment
                    }
                    self.advance(); // Consume comment char
                }
                self.advance(); // Move to next char after comment
                return self.next_token(); // Recurse to get next token
            }
            Some('(') => Token::LParen,
            Some(')') => Token::RParen,
            Some(';') => Token::Semicolon,
            Some(',') => Token::Comma,  // Added
            Some('{') => Token::LBrace, // Added
            Some('}') => Token::RBrace, // Added

            Some(ch) if is_identifier_start(ch) => {
                let ident = self.read_identifier();
                return match ident.as_str() {
                    // Return directly
                    "let" => Token::Let,
                    "fun" => Token::Fun, // Added
                    _ => Token::Identifier(ident),
                };
            }

            Some(ch)
                if ch.is_digit(10)
                    || (ch == '.' && self.peek_char.map_or(false, |pc| pc.is_digit(10))) =>
            {
                return self.read_number(); // Return directly
            }

            Some(ch) => Token::Illegal(ch),
            None => Token::Eof,
        };

        self.advance(); // Move ahead for single-char tokens
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

    // Updated read_number to handle floats like .5 correctly using peek
    fn read_number(&mut self) -> Token {
        let mut number_str = String::new();
        let mut has_decimal = false;

        // Loop while the current character is a digit or a single decimal point
        while let Some(ch) = self.current_char {
            if ch.is_digit(10) {
                number_str.push(ch);
            } else if ch == '.' {
                if has_decimal {
                    break;
                } // Only one decimal point
                // Check if the next char is a digit before accepting the '.'
                // Allow number to end with '.' for now, parse handles validation if needed
                number_str.push(ch);
                has_decimal = true;
            } else {
                break; // Not a digit or allowed '.'
            }
            self.advance(); // Consume the character
        }

        // Note: current_char is now positioned *after* the number

        match number_str.parse::<f64>() {
            Ok(num) => Token::Number(num),
            // Handle cases like just "." or "1.2.3" which might parse partially or fail
            // If parse fails, the first char was likely the issue (or sequence like "..")
            Err(_) => Token::Illegal(number_str.chars().next().unwrap_or('?')), // Simplified error
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
            Token::Number(10.0),
            Token::Semicolon,
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Assign,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::Number(5.0),
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
            Token::Number(0.5),
            Token::Plus,
            Token::Number(1.0),
            Token::Eof,
        ];
        for expected_token in tokens {
            let actual_token = lexer.next_token();
            assert_eq!(actual_token, expected_token);
        }
    }

    #[test]
    fn test_float_parsing_trailing_dot_error() {
        // Assuming "5." is not a valid number on its own for now
        // (Some lexers might allow it, parsing would handle it)
        let input = "5.";
        let mut lexer = Lexer::new(input);
        // Our current parse will likely succeed `5.`.parse::<f64>() ok -> 5.0
        // If we wanted `5.` to be illegal, read_number needs adjustment
        // Let's keep it simple and assume `5.` parses as 5.0 for now.
        assert_eq!(lexer.next_token(), Token::Number(5.0));
        assert_eq!(lexer.next_token(), Token::Eof);

        // Test case for an clearly illegal sequence starting with '.'
        let input = " . ";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next_token(), Token::Illegal('.'));
    }

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
            Token::Number(10.0),
            Token::Semicolon,
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Assign,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::Number(5.0),
            Token::Semicolon,
            Token::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token(), expected);
        }
    }
}

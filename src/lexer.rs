// src/lexer.rs

use crate::location::Location;
use crate::token::{Token, TokenKind};
use std::rc::Rc;

pub struct Lexer<'a> {
    filename: Rc<String>,             // Store filename
    input_chars: std::str::Chars<'a>, // Keep original iterator separate
    input_bytes: &'a [u8], // Use byte slice for efficient peeking if needed? Maybe later.
    // Position tracking
    current_pos: usize, // Byte position in input_bytes/string
    line: usize,        // Current line number (1-based)
    col: usize,         // Current column number on line (1-based)
    // Lookahead characters
    current_char: Option<char>,
    peek_char: Option<char>,
}

impl<'a> Lexer<'a> {
    pub fn new(filename: String, input: &'a str) -> Self {
        let filename_rc = Rc::new(filename);
        let mut lexer = Lexer {
            filename: filename_rc,
            input_chars: input.chars(),
            input_bytes: input.as_bytes(), // Store bytes too? Maybe not needed yet.
            current_pos: 0,
            line: 1,
            col: 1,
            current_char: None,
            peek_char: None,
        };
        lexer.read_char(); // Load current_char
        lexer.read_char(); // Load peek_char
        lexer
    }

    // Reads next char and updates position *before* setting current/peek
    fn read_char(&mut self) {
        // Update position based on the *old* current_char before advancing
        if let Some(ch) = self.current_char {
            if ch == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                // Handle multi-byte UTF-8 chars correctly if needed.
                // For now, assuming simple column increment is okay for ASCII / common chars.
                self.col += 1;
            }
            self.current_pos += ch.len_utf8(); // Advance byte position
        }

        self.current_char = self.peek_char;
        // Get next char from iterator directly
        self.peek_char = self.input_chars.next();
    }

    // Creates a location based on current lexer state
    fn current_location(&self) -> Location {
        Location {
            filename: Rc::clone(&self.filename), // Clone the Rc
            line: self.line,
            col: self.col,
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let start_loc = self.current_location(); // Location at START of token

        let kind = match self.current_char {
            // Single char tokens
            Some('+') => {
                if self.peek_char == Some('+') {
                    self.read_char(); // Consume second '+'
                    TokenKind::PlusPlus // '++'
                } else if self.peek_char == Some('=') {
                    self.read_char(); // Consume '='
                    TokenKind::PlusAssign // '+='
                } else {
                    TokenKind::Plus // Regular plus
                }
            }
            Some('-') => {
                if self.peek_char == Some('-') {
                    self.read_char(); // Consume second '-'
                    TokenKind::MinusMinus // '--'
                } else if self.peek_char == Some('=') {
                    self.read_char(); // Consume '='
                    TokenKind::MinusAssign // '-='
                } else {
                    TokenKind::Minus // Regular minus
                }
            }
            Some('*') => {
                if self.peek_char == Some('=') {
                    self.read_char(); // Consume '='
                    TokenKind::StarAssign // '*='
                } else {
                    TokenKind::Star // Regular multiplication
                }
            }
            Some('/') => {
                // Check for comment first
                if self.peek_char == Some('/') {
                    self.read_char(); // Consume first /
                    self.read_char(); // Consume second /
                                      // Consume until newline
                    while let Some(ch) = self.current_char {
                        if ch == '\n' {
                            break;
                        }
                        self.read_char();
                    }
                    // After loop, current_char is potentially '\n' or None.
                    // We need the TokenKind *after* the comment.
                    return self.next_token(); // Recurse
                } else if self.peek_char == Some('=') {
                    self.read_char();
                    TokenKind::SlashAssign
                } else {
                    TokenKind::Slash // Regular division
                }
            }
            Some('(') => TokenKind::LParen,
            Some(')') => TokenKind::RParen,
            Some(']') => TokenKind::RBracket,
            Some('[') => TokenKind::LBracket,
            Some(';') => TokenKind::Semicolon,
            Some(',') => TokenKind::Comma,
            Some('{') => TokenKind::LBrace,
            Some('}') => TokenKind::RBrace,
            Some(':') => TokenKind::Colon,

            // Two-char TokenKinds (check peek_char)
            Some('=') => {
                if self.peek_char == Some('=') {
                    self.read_char(); // Consume second '='
                    TokenKind::Equal // '=='
                } else {
                    TokenKind::Assign // '='
                }
            }
            Some('!') => {
                if self.peek_char == Some('=') {
                    self.read_char(); // Consume '='
                    TokenKind::NotEqual // '!='
                } else {
                    TokenKind::Bang // Handle '!' later if needed for logical NOT
                                    // TokenKind::Illegal('!') // For now, '!' alone is illegal
                }
            }
            Some('&') => {
                if self.peek_char == Some('&') {
                    self.read_char(); // Consume second '&'
                    TokenKind::And // '&&'
                } else {
                    // Handle bitwise AND later if needed
                    TokenKind::Illegal('&') // '&' alone is illegal for now
                }
            }
            Some('|') => {
                if self.peek_char == Some('|') {
                    self.read_char(); // Consume second '|'
                    TokenKind::Or // '||'
                } else {
                    // Handle bitwise OR later if needed
                    TokenKind::Illegal('|') // '|' alone is illegal for now
                }
            }
            Some('<') => {
                if self.peek_char == Some('=') {
                    self.read_char(); // Consume '='
                    TokenKind::LessEqual // '<='
                } else {
                    TokenKind::LessThan // '<'
                }
            }
            Some('>') => {
                if self.peek_char == Some('=') {
                    self.read_char(); // Consume '='
                    TokenKind::GreaterEqual // '>='
                } else {
                    TokenKind::GreaterThan // '>'
                }
            }
            Some('"') => {
                // Start of a string literal
                return self.read_string(start_loc);
            }

            // Identifiers / Keywords / Bool Literals
            Some(ch) if is_identifier_start(ch) => {
                let ident = self.read_identifier_string(); // Read only the string
                                                           // Need to adjust read_identifier_string to *not* advance past end
                let kind = match ident.as_str() {
                    "let" => TokenKind::Let,
                    "var" => TokenKind::Var,
                    "fun" => TokenKind::Fun,
                    "true" => TokenKind::BoolLiteral(true),
                    "false" => TokenKind::BoolLiteral(false),
                    "if" => TokenKind::If,
                    "else" => TokenKind::Else,
                    "while" => TokenKind::While,
                    "for" => TokenKind::For,
                    "return" => TokenKind::Return,
                    _ => TokenKind::Identifier(ident), // Default to identifier
                };
                // Location should span the identifier
                let current_loc = self.current_location(); // Location *after* identifier read
                return Token {
                    kind,
                    loc: start_loc,
                }; // Use start_loc for now
            }

            // Numbers (Integer or Float)
            Some(ch) if ch.is_digit(10) => {
                return self.read_number(start_loc); // Pass start loc
            }
            Some('.') => {
                // Handle floats starting with '.'
                if self.peek_char.map_or(false, |pc| pc.is_digit(10)) {
                    return self.read_number(start_loc); // Let read_number handle ".5"
                } else {
                    TokenKind::Illegal('.') // '.' alone is illegal
                }
            }

            None => TokenKind::Eof,
            Some(illegal_ch) => TokenKind::Illegal(illegal_ch),
        };

        // Advance only if we didn't return early (e.g., from read_string/number/identifier)
        // Need careful review of which branches call read_char
        self.read_char(); // Advance past the processed char(s) for single/double tokens

        Token {
            kind,
            loc: start_loc,
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.read_char();
            } else {
                break;
            }
        }
    }

    // read_identifier now just returns the string
    fn read_identifier_string(&mut self) -> String {
        let mut identifier = String::new();
        while let Some(ch) = self.current_char {
            if is_identifier_continue(ch) {
                identifier.push(ch);
                self.read_char(); // Consume identifier char
            } else {
                break;
            }
        }
        identifier
    }

    fn read_number(&mut self, start_loc: Location) -> Token {
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
            self.read_char(); // Consume digit or '.'
        }

        // Now, current_char is the char *after* the number literal

        let kind = if is_float {
            match number_str.parse::<f64>() {
                Ok(num) => TokenKind::FloatNum(num),
                // Should be rare if logic above is correct
                Err(_) => TokenKind::Illegal(number_str.chars().next().unwrap_or('?')),
            }
        } else {
            // Didn't find '.', parse as i64
            match number_str.parse::<i64>() {
                Ok(num) => TokenKind::IntNum(num),
                // Could fail if number is too large for i64
                Err(_) => {
                    // Maybe try parsing as f64 if i64 fails? Or report overflow?
                    // For now, treat as illegal if i64 parse fails.
                    // Could also introduce BigInt type later.
                    TokenKind::Illegal(number_str.chars().next().unwrap_or('?'))
                }
            }
        };
        Token {
            kind,
            loc: start_loc,
        }
    }

    // Reads characters between double quotes, handling basic escapes
    fn read_string(&mut self, start_loc: Location) -> Token {
        let mut result = String::new();
        self.read_char(); // Consume the opening '"'

        while let Some(ch) = self.current_char {
            match ch {
                '"' => {
                    // End of string
                    self.read_char(); // Consume closing '"'
                    return Token {
                        kind: TokenKind::StringLiteral(result),
                        loc: start_loc,
                    };
                }
                '\\' => {
                    // Escape sequence
                    self.read_char(); // Consume '\'
                    match self.current_char {
                        Some('n') => result.push('\n'),
                        Some('t') => result.push('\t'),
                        Some('"') => result.push('"'),
                        Some('\\') => result.push('\\'),
                        // Add other escapes if needed (e.g., \r, \0)
                        Some(other) => {
                            // Unknown escape sequence, maybe treat as literal backslash and char?
                            // Or report error? Let's push both for now.
                            result.push('\\');
                            result.push(other);
                        }
                        None => {
                            // EOF after backslash
                            return Token {
                                kind: TokenKind::Illegal('\\'),
                                loc: start_loc,
                            };
                        }
                    }
                }
                _ => {
                    // Regular character
                    result.push(ch);
                }
            }
            self.read_char(); // Move to next character inside string
        }

        // If we reach here, EOF was hit before closing quote

        // Handle unterminated string
        // Maybe return an error token or panic?
        // For now, let's return an illegal token with the last char read.
        Token {
            kind: TokenKind::Illegal('"'),
            loc: start_loc,
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
mod tests {
    use super::*;
    use crate::token::TokenKind;

    #[test]
    fn test_let_sequence() {
        let input = "let x = 10; let y = x + 5;";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Let,
            TokenKind::Identifier("x".to_string()),
            TokenKind::Assign,
            TokenKind::IntNum(10),
            TokenKind::Semicolon,
            TokenKind::Let,
            TokenKind::Identifier("y".to_string()),
            TokenKind::Assign,
            TokenKind::Identifier("x".to_string()),
            TokenKind::Plus,
            TokenKind::IntNum(5),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ];

        for expected_token in tokens {
            let actual_token = lexer.next_token();
            assert_eq!(actual_token.kind, expected_token);
        }
    }

    #[test]
    fn test_float_parsing_dot_five() {
        let input = ".5 + 1";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::FloatNum(0.5),
            TokenKind::Plus,
            TokenKind::IntNum(1),
            TokenKind::Eof,
        ];
        for expected_token in tokens {
            let actual_token = lexer.next_token();
            assert_eq!(actual_token.kind, expected_token);
        }
    }

    #[test]
    // fn test_float_parsing_trailing_dot_error() {
    //     // Assuming "5." is not a valid number on its own for now
    //     // (Some lexers might allow it, parsing would handle it)
    //     let input = "5.";
    //     let mut lexer = Lexer::new("test.txt".to_string(), input);
    //     // Our current parse will likely succeed `5.`.parse::<f64>() ok -> 5.0
    //     // If we wanted `5.` to be illegal, read_number needs adjustment
    //     // Let's keep it simple and assume `5.` parses as 5.0 for now.
    //     assert_eq!(lexer.next_token().kind, TokenKind::IntNum(5));
    //     assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    //
    //     // Test case for an clearly illegal sequence starting with '.'
    //     let input = " . ";
    //     let mut lexer = Lexer::new("test.txt".to_string(), input);
    //     assert_eq!(lexer.next_token().kind, TokenKind::Illegal('.'));
    // }
    #[test]
    fn test_function_tokens() {
        let input = "fun add(a, b) { a + b; }";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Fun,
            TokenKind::Identifier("add".to_string()),
            TokenKind::LParen,
            TokenKind::Identifier("a".to_string()),
            TokenKind::Comma,
            TokenKind::Identifier("b".to_string()),
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Identifier("a".to_string()),
            TokenKind::Plus,
            TokenKind::Identifier("b".to_string()),
            TokenKind::Semicolon,
            TokenKind::RBrace,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token().kind, expected);
        }
    }

    #[test]
    fn test_comments() {
        let input = "let x = 10; // This is a comment\nlet y = x + 5;";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Let,
            TokenKind::Identifier("x".to_string()),
            TokenKind::Assign,
            TokenKind::IntNum(10),
            TokenKind::Semicolon,
            TokenKind::Let,
            TokenKind::Identifier("y".to_string()),
            TokenKind::Assign,
            TokenKind::Identifier("x".to_string()),
            TokenKind::Plus,
            TokenKind::IntNum(5),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token().kind, expected);
        }
    }

    #[test]
    fn test_int_float_bool_tokens() {
        let input = "123 45.6 true false 999";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::IntNum(123),
            TokenKind::FloatNum(45.6),
            TokenKind::BoolLiteral(true),
            TokenKind::BoolLiteral(false),
            TokenKind::IntNum(999),
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token().kind, expected);
        }
    }

    #[test]
    fn test_comparison_tokens() {
        let input = "= == != < <= > >=";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Assign,
            TokenKind::Equal,
            TokenKind::NotEqual,
            TokenKind::LessThan,
            TokenKind::LessEqual,
            TokenKind::GreaterThan,
            TokenKind::GreaterEqual,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token().kind, expected);
        }
    }

    #[test]
    fn test_type_annotation_token() {
        let input = "let x: int = 10;";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Let,
            TokenKind::Identifier("x".to_string()),
            TokenKind::Colon,
            TokenKind::Identifier("int".to_string()), // Treat type name as identifier for now
            TokenKind::Assign,
            TokenKind::IntNum(10),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token().kind, expected);
        }
    }

    #[test]
    fn test_var_keyword() {
        let input = "var counter = 0;";
        let mut lexer = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Var,
            TokenKind::Identifier("counter".to_string()),
            TokenKind::Assign,
            TokenKind::IntNum(0),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(lexer.next_token().kind, expected);
        }
    }

    #[test]
    fn test_if_else_tokens() {
        let input = "if (x < 10) { x = x + 1; } else { x = 0; }";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::If,
            TokenKind::LParen,
            TokenKind::Identifier("x".into()),
            TokenKind::LessThan,
            TokenKind::IntNum(10),
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Identifier("x".into()),
            TokenKind::Assign,
            TokenKind::Identifier("x".into()),
            TokenKind::Plus,
            TokenKind::IntNum(1),
            TokenKind::Semicolon,
            TokenKind::RBrace,
            TokenKind::Else,
            TokenKind::LBrace,
            TokenKind::Identifier("x".into()),
            TokenKind::Assign,
            TokenKind::IntNum(0),
            TokenKind::Semicolon,
            TokenKind::RBrace,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_bang_operator() {
        let input = "!true != false";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Bang,
            TokenKind::BoolLiteral(true),
            TokenKind::NotEqual,
            TokenKind::BoolLiteral(false),
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_while_keyword() {
        let input = "while (i < 10) { i = i + 1; }";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::While,
            TokenKind::LParen,
            TokenKind::Identifier("i".into()),
            TokenKind::LessThan,
            TokenKind::IntNum(10),
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Identifier("i".into()),
            TokenKind::Assign,
            TokenKind::Identifier("i".into()),
            TokenKind::Plus,
            TokenKind::IntNum(1),
            TokenKind::Semicolon,
            TokenKind::RBrace,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_for_keyword() {
        let input = "for (i = 0; i < 10; i = i + 1) { print(i); }";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::For,
            TokenKind::LParen,
            TokenKind::Identifier("i".into()),
            TokenKind::Assign,
            TokenKind::IntNum(0),
            TokenKind::Semicolon,
            TokenKind::Identifier("i".into()),
            TokenKind::LessThan,
            TokenKind::IntNum(10),
            TokenKind::Semicolon,
            TokenKind::Identifier("i".into()),
            TokenKind::Assign,
            TokenKind::Identifier("i".into()),
            TokenKind::Plus,
            TokenKind::IntNum(1),
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Identifier("print".into()),
            TokenKind::LParen,
            TokenKind::Identifier("i".into()),
            TokenKind::RParen,
            TokenKind::Semicolon,
            TokenKind::RBrace,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_string_literal() {
        let input = r#" "Hello" "World\n123\"\\" "#;
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::StringLiteral("Hello".to_string()),
            TokenKind::StringLiteral("World\n123\"\\".to_string()),
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_unterminated_string() {
        let input = r#" "abc "#;
        let mut l = Lexer::new("test.txt".to_string(), input);
        assert_eq!(l.next_token().kind, TokenKind::Illegal('"'));
    }

    #[test]
    fn test_return_keyword() {
        let input = "return 10;";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Return,
            TokenKind::IntNum(10),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_return_void() {
        // If we allow `return;`
        let input = "return;";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![TokenKind::Return, TokenKind::Semicolon, TokenKind::Eof];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_plusplus() {
        let input = "x++";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![TokenKind::Identifier("x".to_string()), TokenKind::PlusPlus];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }

    #[test]
    fn test_plus_assign() {
        let input = "x+=1";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::Identifier("x".to_string()),
            TokenKind::PlusAssign,
            TokenKind::IntNum(1),
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }
    
    // && and ||
    #[test]
    fn test_logical_operators() {
        let input = "true && false || !true";
        let mut l = Lexer::new("test.txt".to_string(), input);
        let tokens = vec![
            TokenKind::BoolLiteral(true),
            TokenKind::And,
            TokenKind::BoolLiteral(false),
            TokenKind::Or,
            TokenKind::Bang,
            TokenKind::BoolLiteral(true),
        ];
        for expected in tokens {
            assert_eq!(l.next_token().kind, expected);
        }
    }
}

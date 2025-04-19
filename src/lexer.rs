use crate::token::Token;
// Import the Token enum from token.rs

pub struct Lexer<'a> {
    input: std::str::Chars<'a>, // Iterator over the characters of the input string
    current_char: Option<char>, // The character we are currently looking at
                                // We could add position tracking (line/column) here later if needed
}

impl<'a> Lexer<'a> {
    // Constructor for the Lexer
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input: input.chars(), // Create a character iterator from the input string slice
            current_char: None,   // Initialize current_char to None
        };
        lexer.advance(); // Load the first character into current_char
        lexer
    }

    // Helper function to advance the iterator and update current_char
    fn advance(&mut self) {
        self.current_char = self.input.next(); // Get the next char, or None if EOF
    }

    // The main function to get the next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace(); // Ignore any leading whitespace

        let token = match self.current_char {
            Some('+') => Token::Plus,
            Some('-') => Token::Minus,
            Some('*') => Token::Star,
            Some('/') => Token::Slash,
            Some('(') => Token::LParen,
            Some(')') => Token::RParen,

            Some(ch) if ch.is_digit(10) || ch == '.' => {
                // If it starts with a digit or a dot, attempt to parse a number
                return self.read_number(); // Return directly as read_number consumes chars
            }

            Some(ch) => Token::Illegal(ch), // Unrecognized character

            None => Token::Eof, // End of input
        };

        self.advance(); // Move to the next character *after* processing the current one
        // (unless read_number already advanced)
        token
    }

    // Helper function to skip whitespace characters
    fn skip_whitespace(&mut self) {
        // Loop while the current character is whitespace
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance(); // If it's whitespace, move to the next character
            } else {
                break; // If it's not whitespace, stop skipping
            }
        }
    }

    // Helper function to read a number (integer or floating point)
    fn read_number(&mut self) -> Token {
        let mut number_str = String::new();
        let mut has_decimal = false;

        // Loop while the current character is a digit or a decimal point
        while let Some(ch) = self.current_char {
            if ch.is_digit(10) {
                number_str.push(ch);
            } else if ch == '.' {
                // Allow only one decimal point
                if has_decimal {
                    // If we already saw a '.', this is likely an error or part of another token
                    // For simplicity now, we stop reading the number here.
                    // A more robust lexer might handle "1.2.3" as Number(1.2) followed by Illegal('.') etc.
                    break;
                }
                number_str.push(ch);
                has_decimal = true;
            } else {
                // If it's neither a digit nor a '.', the number ends here
                break;
            }
            self.advance(); // Consume the character and move to the next one
        }

        // Try to parse the collected string into an f64
        match number_str.parse::<f64>() {
            Ok(num) => Token::Number(num),
            Err(_) => {
                // If parsing fails (e.g., just a "." was read), treat it as illegal.
                // We could return the first char of number_str if needed.
                // For simplicity, let's just return a generic Illegal token for now.
                // A better approach might involve more nuanced error reporting or
                // backtracking if the '.' was meant for something else.
                // Since read_number consumes characters, we need to be careful.
                // If number_str is just ".", it's an illegal start here.
                // If number_str is empty (shouldn't happen due to initial check), also illegal.
                if number_str.is_empty() {
                    // This case should technically not be reachable due to the entry condition
                    // in next_token, but handle defensively.
                    Token::Illegal('?') // Or some other indicator
                } else {
                    // Return the first character as Illegal, although we've consumed more.
                    // This is a simplification; proper error recovery is complex.
                    Token::Illegal(number_str.chars().next().unwrap_or('?'))
                }
            }
        }
        // Note: We don't call self.advance() at the end here because the loop
        // inside read_number already advanced past the last character of the number.
        // The self.current_char now holds the character *after* the number.
    }
}

// We'll add tests later

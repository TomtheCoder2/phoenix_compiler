#[derive(Debug, PartialEq, Clone)] // Added Clone
pub enum Token {
    // Special Tokens
    Illegal(char), // Represents a character we don't understand
    Eof,           // Represents the end of the input file

    // Literals
    Number(f64), // Represents a floating-point number, e.g., 123.45

    // Operators
    Plus,        // '+'
    Minus,       // '-'
    Star,        // '*'
    Slash,       // '/'

    // Delimiters
    LParen,      // '('
    RParen,      // ')'

    // Keywords (We'll add these later)

    // Identifiers (We'll add these later)
}
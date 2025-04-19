// src/token.rs

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Special Tokens
    Illegal(char),
    Eof,

    // Literals
    Number(f64),

    // Operators
    Plus,        // '+'
    Minus,       // '-'
    Star,        // '*'
    Slash,       // '/'
    Assign,      // '=' // Added

    // Delimiters
    LParen,      // '('
    RParen,      // ')'

    // Keywords
    Let,         // 'let' // Added
    In,          // 'in'  // Added

    // Identifiers
    Identifier(String), // e.g., "my_variable" // Added
}
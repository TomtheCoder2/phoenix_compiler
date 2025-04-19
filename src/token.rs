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
    Assign,      // '='

    // Delimiters
    LParen,      // '('
    RParen,      // ')'
    Semicolon,   // ';' // Added

    // Keywords
    Let,         // 'let'
    // In,       // 'in' // Removed (or keep for future functional constructs?) Let's remove for now.

    // Identifiers
    Identifier(String), // e.g., "my_variable"
}
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
    Semicolon,   // ';'
    Comma,       // ',' // Added
    LBrace,      // '{' // Added
    RBrace,      // '}' // Added

    // Keywords
    Let,         // 'let'
    Fun,         // 'fun' // Added

    // Identifiers
    Identifier(String), // e.g., "my_variable"
}
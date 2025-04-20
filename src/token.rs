// src/token.rs

use crate::types::Type;
// Import our new Type enum

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Special Tokens
    Illegal(char),
    Eof,

    // Literals
    FloatNum(f64),     // Renamed from Number
    IntNum(i64),       // Added for integers
    BoolLiteral(bool), // Added for true/false

    // Operators
    Plus,  // '+'
    Minus, // '-'
    Star,  // '*'
    Slash, // '/'
    Assign, // '='

    // Comparison Operators (Added)
    LessThan,    // '<'
    GreaterThan, // '>'
    Equal,       // '=='
    NotEqual,    // '!='
    LessEqual,   // '<='
    GreaterEqual,// '>='

    // Logical Operators (Add later?)
    Bang, // '!' (Negation)
    // And,  // '&&'
    // Or,   // '||'

    // Delimiters
    LParen,    // '('
    RParen,    // ')'
    Semicolon, // ';'
    Comma,     // ','
    LBrace,    // '{'
    RBrace,    // '}'
    Colon,     // ':' //  for type annotations (e.g., let x: int)

    // Keywords
    Let, // 'let'
    Var, // 'var' (mutable binding) 
    Fun, // 'fun'
    True, // 'true' 
    False, // 'false' 
    If, // 'if'
    Else, // 'else'
    While, // 'while' 
    For, // 'for'
    // Type Keywords (optional, could use identifiers)
    // TypeInt, // 'int'
    // TypeFloat, // 'float'
    // TypeBool, // 'bool'

    // Identifiers
    Identifier(String),
}

// Helper to maybe parse type names (optional, can rely on identifiers)
pub fn keyword_to_type(ident: &str) -> Option<Type> {
    match ident {
        "int" => Some(Type::Int),
        "float" => Some(Type::Float),
        "bool" => Some(Type::Bool),
        _ => None,
    }
}
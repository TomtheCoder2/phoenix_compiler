// src/TokenKind.rs

use crate::location::Location;
use crate::types::Type;
// Import our new Type enum

// The actual token data structure passed around
#[derive(Debug, Clone)] // Remove PartialEq auto-derive for now
pub struct Token {
    pub kind: TokenKind,
    pub loc: Location,
}

// Custom PartialEq to ignore location for comparisons in parser etc.
impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind // Compare only the kind
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    // Special TokenKinds
    Illegal(char),
    Eof,

    // Literals
    FloatNum(f64),         // Renamed from Number
    IntNum(i64),           // Added for integers
    BoolLiteral(bool),     // Added for true/false
    StringLiteral(String), // Added for string literals

    // Operators
    Plus,        // '+'
    PlusAssign,  // '+='
    PlusPlus,    // '++'
    Minus,       // '-'
    MinusAssign, // '-='
    MinusMinus,  // '--'
    Star,        // '*'
    StarAssign,  // '*='
    Slash,       // '/'
    SlashAssign, // '/='
    Assign,      // '='

    // Comparison Operators (Added)
    LessThan,     // '<'
    GreaterThan,  // '>'
    Equal,        // '=='
    NotEqual,     // '!='
    LessEqual,    // '<='
    GreaterEqual, // '>='

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
    Colon,     // ':' // for type annotations (e.g., let x: int)
    LBracket,  // '[' // Added for indexing & literals
    RBracket,  // ']' // Added

    // Keywords
    Let,    // 'let'
    Var,    // 'var' (mutable binding)
    Fun,    // 'fun'
    True,   // 'true'
    False,  // 'false'
    If,     // 'if'
    Else,   // 'else'
    While,  // 'while'
    For,    // 'for'
    Return, // 'return'
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
        "void" => Some(Type::Void),
        _ => None,
    }
}

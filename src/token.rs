// src/TokenKind.rs

use std::fmt;
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
    And,  // '&&'
    Or,   // '||'

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

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Illegal(c) => write!(f, "Illegal({})", c),
            TokenKind::Eof => write!(f, "EOF"),
            TokenKind::FloatNum(n) => write!(f, "FloatNum({})", n),
            TokenKind::IntNum(n) => write!(f, "IntNum({})", n),
            TokenKind::BoolLiteral(b) => write!(f, "BoolLiteral({})", b),
            TokenKind::StringLiteral(s) => write!(f, "StringLiteral({})", s),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::PlusAssign => write!(f, "+="),
            TokenKind::PlusPlus => write!(f, "++"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::MinusAssign => write!(f, "-="),
            TokenKind::MinusMinus => write!(f, "--"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::StarAssign => write!(f, "*="),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::SlashAssign => write!(f, "/="),
            TokenKind::Assign => write!(f, "="),
            TokenKind::LessThan => write!(f, "<"),
            TokenKind::GreaterThan => write!(f, ">"),
            TokenKind::Equal => write!(f, "=="),
            TokenKind::NotEqual => write!(f, "!="),
            TokenKind::LessEqual => write!(f, "<="),
            TokenKind::GreaterEqual => write!(f, ">="),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::Fun => write!(f, "fun"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::While => write!(f, "while"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Identifier(s) => write!(f, "Identifier({})", s),
            // TokenKind::TypeInt => write!(f, "int"),
            // TokenKind::TypeFloat => write!(f, "float"),
            // TokenKind::TypeBool => write!(f, "bool"),
        }
    }
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

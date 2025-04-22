use crate::location::Span;
use crate::types::Type;

// src/ast.rs
pub type NumberType = f64;

// Program is still a list of statements
#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub statements: Vec<Statement>, // Vec of Statement structs
    pub span: Span,                 // Added span covering the whole program/block
}

// Wrapper Structs
#[derive(Debug, PartialEq, Clone)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span, // Span covering the whole statement
}

pub fn defs(statement_kind: StatementKind) -> Statement {
    Statement {
        kind: statement_kind,
        span: Span::default(),
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span, // Span covering the whole expression
}

/// Creates a new `Expression` from the given `ExpressionKind`.
///
/// # Arguments
///
/// * `expr` - The kind of expression to wrap in an `Expression` struct.
///
/// # Returns
///
/// An `Expression` struct containing the provided `ExpressionKind` and a default `Span`.
///
/// # Example
///
/// ```rust
/// use toylang_compiler::ast::{ExpressionKind, def};
/// use toylang_compiler::location::Span;
///
/// let expr_kind = ExpressionKind::IntLiteral(42);
/// let expression = def(expr_kind);
/// assert_eq!(expression.kind, ExpressionKind::IntLiteral(42));
/// assert_eq!(expression.span, Span::default());
/// ```
pub fn def(expr: ExpressionKind) -> Expression {
    Expression {
        kind: expr,
        span: Span::default(), // Default span for now
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum StatementKind {
    // Renamed from Statement
    LetBinding {
        name: String,
        type_ann: Option<Type>,
        value: Expression,
    }, // Value is now Expression struct
    VarBinding {
        name: String,
        type_ann: Option<Type>,
        value: Expression,
    },
    ExpressionStmt(Expression), // Holds Expression struct
    FunctionDef {
        name: String,
        params: Vec<(String, Option<Type>)>,
        return_type_ann: Option<Type>,
        body: Program,
    }, // Body is Program struct
    IfStmt {
        condition: Expression,
        then_branch: Program,
        else_branch: Option<Program>,
    }, // Holds Expression/Program structs
    WhileStmt {
        condition: Expression,
        body: Program,
    },
    ForStmt {
        initializer: Option<Expression>,
        condition: Option<Expression>,
        increment: Option<Expression>,
        body: Program,
    },
    ReturnStmt {
        value: Option<Expression>,
    },
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum UnaryOperator {
    Negate, // Arithmetic negation (-)
    Not,    // Logical not (!)
}

#[derive(Debug, PartialEq, Clone)]
pub enum ExpressionKind {
    // Renamed from Expression
    FloatLiteral(f64),
    IntLiteral(i64),
    BoolLiteral(bool),
    StringLiteral(String),
    Variable(String),
    Assignment {
        target: String,
        value: Box<Expression>,
    }, // Holds Box<Expression> struct
    IfExpr {
        condition: Box<Expression>,
        then_branch: Box<Expression>,
        else_branch: Box<Expression>,
    },
    Block {
        statements: Vec<Statement>,
        final_expression: Option<Box<Expression>>,
    }, // Holds Vec<Statement>, Box<Expression> structs
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    ComparisonOp {
        op: ComparisonOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },
}

// Separate enums for operator types
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ComparisonOperator {
    LessThan,
    GreaterThan,
    Equal,
    NotEqual,
    LessEqual,
    GreaterEqual,
}

// #[derive(Debug, PartialEq, Clone, Copy)]
// pub enum UnaryOperator { Not } // Logical Not

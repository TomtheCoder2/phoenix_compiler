use crate::types::Type;

// src/ast.rs
pub type NumberType = f64;

// Program is still a list of statements
#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    LetBinding {
        name: String,
        // Optional type annotation (parsed but maybe not checked yet)
        type_ann: Option<Type>,
        value: Expression,
    },
    // Mutable binding
    VarBinding { 
        name: String,
        type_ann: Option<Type>,
        value: Expression,
    },
    ExpressionStmt(Expression),
    FunctionDef {
        name: String,
        // Parameters now need optional type annotations
        params: Vec<(String, Option<Type>)>,
        // Optional return type annotation
        return_type_ann: Option<Type>,
        body: Box<Program>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    // Literals
    FloatLiteral(f64), // Renamed
    IntLiteral(i64),   // Added
    BoolLiteral(bool), // Added

    // Variable & Call
    Variable(String),
    // Assignment: target = value
    Assignment { // Added
        target: String, // Name of variable being assigned to
        value: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },

    // Operators
    BinaryOp {
        op: BinaryOperator, // Keep current arithmetic ops
        left: Box<Expression>,
        right: Box<Expression>,
    },
    ComparisonOp {
        // Added
        op: ComparisonOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    // UnaryOp { op: UnaryOperator, operand: Box<Expression> }, // Add later for '!' etc.
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

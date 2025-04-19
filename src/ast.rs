// src/ast.rs

pub type NumberType = f64;

// Program is now a list of statements
#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub statements: Vec<Statement>,
}

// Represents different kinds of statements
#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    // Example: let x = 5 * 2;
    LetBinding {
        name: String,
        value: Expression, // Keep value as Expression
    },
    // Example: x + 1; (Evaluated for value, potentially for side effects later)
    ExpressionStmt(Expression),
}

// Expressions remain largely the same, but Let is removed from here
#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    NumberLiteral(NumberType),
    Variable(String),
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    // Let { ... } // Removed from Expression enum
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}
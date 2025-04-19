// src/ast.rs

pub type NumberType = f64;

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    NumberLiteral(NumberType),
    Variable(String), // Added: Represents using a variable, e.g., 'x'
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Let { // Added: Represents 'let name = value in body'
        name: String,          // Variable name being bound
        value: Box<Expression>,// The expression assigned to the variable
        body: Box<Expression>, // The expression where the binding is valid
    },
    // Maybe add FunctionCall later
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}
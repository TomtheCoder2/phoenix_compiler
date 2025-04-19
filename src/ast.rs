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
        value: Expression,
    },
    ExpressionStmt(Expression),
    // Added: Represents 'fun name(params...) { body }'
    FunctionDef {
        name: String,        // Function name
        params: Vec<String>, // Parameter names
        body: Box<Program>, // Use Box<Program> for recursive structure, body is a sequence of stmts
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    NumberLiteral(NumberType),
    Variable(String),
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    // Added: Represents 'func_name(arg1, arg2, ...)'
    FunctionCall {
        name: String,          // Function name being called
        args: Vec<Expression>, // Argument expressions
    },
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

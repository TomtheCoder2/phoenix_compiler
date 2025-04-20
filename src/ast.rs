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
    IfStmt {
        condition: Expression,     // Condition is still an expression
        then_branch: Box<Program>, // Block of statements
        // Else branch is now optional
        else_branch: Option<Box<Program>>,
    },
    WhileStmt {
        condition: Expression, // Condition must evaluate to bool
        body: Box<Program>,    // Block of statements to repeat
    },
    // Added: for (init; cond; incr) { body }
    ForStmt {
        // Initializer: Can be variable declaration or just expression(s)
        // Let's allow only Expression for simplicity now (e.g. `i=0`, not `var i = 0`)
        // Use Option<Expression> - maybe Boxed? Let's box it.
        initializer: Option<Box<Expression>>,
        // Condition: Evaluated before each iteration
        condition: Option<Expression>, // No Box needed if direct Expression
        // Increment: Evaluated after each iteration
        increment: Option<Box<Expression>>,
        // Body: Executed each iteration
        body: Box<Program>,
    },
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum UnaryOperator {
    Negate, // Arithmetic negation (-)
    Not,    // Logical not (!)
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    // Literals
    FloatLiteral(f64), // Renamed
    IntLiteral(i64),
    BoolLiteral(bool),
    StringLiteral(String), 

    // Variable & Call
    Variable(String),
    // Assignment: target = value
    Assignment {
        // Added
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
    IfExpr {
        // Renamed from If from previous chapter
        condition: Box<Expression>,
        then_branch: Box<Expression>, // Use Program for blocks
        else_branch: Box<Expression>, // Mandatory
    },
    // Block used as an expression: { stmt; stmt; final_expr }
    Block {
        statements: Vec<Statement>,
        // Optional final expression determines the block's value
        final_expression: Option<Box<Expression>>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
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

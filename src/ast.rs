// Using f64 for now, consistent with the lexer
pub type NumberType = f64;

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    NumberLiteral(NumberType),
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>, // Box<T> allocates the value on the heap -> against stack overflow, because box is a pointer
        right: Box<Expression>,
    },
    // We can add more variants later, like UnaryOp, Variable, FunctionCall, etc.
}

#[derive(Debug, PartialEq, Clone, Copy)] // Use Copy for simple enums
pub enum BinaryOperator {
    Add,      // +
    Subtract, // -
    Multiply, // *
    Divide,   // /
}

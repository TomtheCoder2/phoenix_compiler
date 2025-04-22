use crate::location::Span;
use crate::types::Type;
use std::cell::RefCell;

// src/ast.rs
pub type NumberType = f64;

// Type Annotation Node (used in Let/Var/Params/Return)
#[derive(Debug, PartialEq, Clone)]
pub struct TypeNode {
    pub kind: TypeNodeKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeNodeKind {
    Simple(String), // "int", "float", "bool", "string"
    Vector(Box<TypeNode>), // vec< T > where T is another TypeNode
                    // Add Function, Tuple etc. later
}

pub fn type_node_to_type(type_node: &TypeNode) -> Type {
    match &type_node.kind {
        TypeNodeKind::Simple(name) => match name.as_str() {
            "int" => Type::Int,
            "float" => Type::Float,
            "bool" => Type::Bool,
            "string" => Type::String,
            "void" => Type::Void,
            _ => panic!("Unknown type: {}", name),
        },
        TypeNodeKind::Vector(inner_type) => {
            let inner = type_node_to_type(inner_type);
            Type::Vector(Box::new(inner))
        }
    }
}

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

impl Statement {
    // convert stmt back to code
    pub fn to_code(&self) -> String {
        todo!()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span, // Span covering the whole expression
    // Added: Stores the type determined by the type checker
    // Use RefCell for interior mutability: allows type checker to set the type
    // on a potentially immutably borrowed AST node during traversal.
    // Default to None initially.
    pub resolved_type: RefCell<Option<Type>>,
}

impl Expression {
    // Helper to create expression node
    pub fn new(kind: ExpressionKind, span: Span) -> Self {
        Expression {
            kind,
            span,
            resolved_type: RefCell::new(None), // Initialize as None
        }
    }

    // // Helper to get the resolved type (immutable borrow)
    // pub fn get_type(&self) -> Option<Type> {
    //     *self.resolved_type.borrow() // Dereference Ref<Option<Type>>
    // }

    // Helper to set the resolved type (mutable borrow)
    pub fn set_type(&self, ty: Type) {
        *self.resolved_type.borrow_mut() = Some(ty);
    }

    // Helper to check if type has been set
    pub fn has_type(&self) -> bool {
        self.resolved_type.borrow().is_some()
    }

    // Helper to convert an expression back to code
    pub fn to_code(&self) -> String {
        match &self.kind {
            ExpressionKind::IntLiteral(value) => value.to_string(),
            ExpressionKind::FloatLiteral(value) => value.to_string(),
            ExpressionKind::BoolLiteral(value) => value.to_string(),
            ExpressionKind::StringLiteral(value) => format!("\"{}\"", value),
            ExpressionKind::Variable(name) => name.clone(),
            ExpressionKind::Assignment { target, value } => {
                format!("{} = {}", target.to_code(), value.to_code())
            }
            ExpressionKind::IfExpr {
                condition,
                then_branch,
                else_branch,
            } => {
                format!(
                    "if ({}) {{{}}} else {{{}}}",
                    condition.to_code(),
                    then_branch.to_code(),
                    else_branch.to_code()
                )
            }
            ExpressionKind::Block {
                statements,
                final_expression,
            } => {
                let statements_code: Vec<String> =
                    statements.iter().map(|stmt| stmt.to_code()).collect();
                let final_expr_code = final_expression
                    .as_ref()
                    .map_or("".to_string(), |expr| expr.to_code());
                format!("{{ {} {} }}", statements_code.join("; "), final_expr_code)
            }
            ExpressionKind::FunctionCall { name, args } => {
                let args_code: Vec<String> = args.iter().map(|arg| arg.to_code()).collect();
                format!("{}({})", name, args_code.join(", "))
            }
            ExpressionKind::BinaryOp { op, left, right } => {
                format!(
                    "{} {} {}",
                    left.to_code(),
                    match op {
                        BinaryOperator::Add => "+",
                        BinaryOperator::Subtract => "-",
                        BinaryOperator::Multiply => "*",
                        BinaryOperator::Divide => "/",
                    },
                    right.to_code()
                )
            }
            ExpressionKind::ComparisonOp { op, left, right } => {
                format!(
                    "{} {} {}",
                    left.to_code(),
                    match op {
                        ComparisonOperator::LessThan => "<",
                        ComparisonOperator::GreaterThan => ">",
                        ComparisonOperator::Equal => "==",
                        ComparisonOperator::NotEqual => "!=",
                        ComparisonOperator::LessEqual => "<=",
                        ComparisonOperator::GreaterEqual => ">=",
                    },
                    right.to_code()
                )
            }
            ExpressionKind::UnaryOp { op, operand } => {
                format!(
                    "{}{}",
                    match op {
                        UnaryOperator::Negate => "-",
                        UnaryOperator::Not => "!",
                    },
                    operand.to_code()
                )
            }
            ExpressionKind::VectorLiteral { elements } => {
                let elements_code: Vec<String> =
                    elements.iter().map(|elem| elem.to_code()).collect();
                format!("[{}]", elements_code.join(", "))
            }
            ExpressionKind::IndexAccess { target, index } => {
                format!("{}[{}]", target.to_code(), index.to_code())
            } // Add more cases as needed
        }
    }
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
        span: Span::default(),             // Default span for now
        resolved_type: RefCell::new(None), // Initialize as None
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum StatementKind {
    // Renamed from Statement
    LetBinding {
        name: String,
        type_ann: Option<TypeNode>,
        value: Expression,
    }, // Value is now Expression struct
    VarBinding {
        name: String,
        type_ann: Option<TypeNode>,
        value: Expression,
    },
    ExpressionStmt(Expression), // Holds Expression struct
    FunctionDef {
        name: String,
        params: Vec<(String, Option<TypeNode>)>,
        return_type_ann: Option<TypeNode>,
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
        target: Box<Expression>,
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
    // Added for Vectors
    VectorLiteral {
        elements: Vec<Expression>, // e.g., [1, 2, 3]
                                   // Element type is inferred or checked later
    },
    IndexAccess {
        target: Box<Expression>, // Expression yielding a vector (or array/string later)
        index: Box<Expression>,  // Expression yielding an integer index
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

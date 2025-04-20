// src/typechecker.rs

use crate::ast::{ComparisonOperator, Expression, Program, Statement, UnaryOperator};
use crate::symbol_table::{FunctionSignature, SymbolInfo, SymbolTable};
use crate::types::Type;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TypeError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    VariableRedefinition(String),
    FunctionRedefinition(String),
    AssignmentToImmutable(String),
    TypeMismatch {
        // General type mismatch
        expected: Type,
        found: Type,
        context: String, // e.g., "if condition", "binary op '+'", "assignment"
    },
    IncorrectArgCount {
        // Arg count mismatch
        func_name: String,
        expected: usize,
        found: usize,
    },
    InvalidOperation {
        // Operation not valid for type(s)
        op: String,        // Operator symbol/name
        type_info: String, // Type(s) involved
    },
    MissingReturnValue(String), // Function doesn't return value matching signature
                                // Add more specific errors
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeError::UndefinedVariable(n) => write!(f, "Undefined variable '{}'", n),
            TypeError::UndefinedFunction(n) => write!(f, "Undefined function '{}'", n),
            TypeError::VariableRedefinition(n) => {
                write!(f, "Variable '{}' redefined in this scope", n)
            }
            TypeError::FunctionRedefinition(n) => write!(f, "Function '{}' redefined", n),
            TypeError::AssignmentToImmutable(n) => {
                write!(f, "Cannot assign to immutable variable '{}'", n)
            }
            TypeError::TypeMismatch {
                expected,
                found,
                context,
            } => write!(
                f,
                "Type mismatch in {}: expected {}, found {}",
                context, expected, found
            ),
            TypeError::IncorrectArgCount {
                func_name,
                expected,
                found,
            } => write!(
                f,
                "Function '{}' called with {} arguments, expected {}",
                func_name, found, expected
            ),
            TypeError::InvalidOperation { op, type_info } => {
                write!(f, "Invalid operation '{}' for type(s) {}", op, type_info)
            }
            TypeError::MissingReturnValue(fname) => write!(
                f,
                "Function '{}' may not return a value in all paths",
                fname
            ),
        }
    }
}

pub struct TypeChecker {
    symbol_table: SymbolTable,
    errors: Vec<TypeError>,
    // Track expected return type for function bodies
    current_function_return_type: Option<Type>,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            symbol_table: SymbolTable::new(),
            errors: Vec::new(),
            current_function_return_type: None,
        }
    }

    /// Main entry point: Check a whole program.
    pub fn check_program(&mut self, program: &Program) -> Result<(), Vec<TypeError>> {
        // TODO: Maybe pre-pass to collect all function signatures first? Allows forward references.
        // For now, functions must be defined before use.

        for statement in &program.statements {
            if let Err(e) = self.check_statement(statement) {
                self.errors.push(e); // Collect error but continue checking if possible
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            // Return cloned errors
            Err(self.errors.clone())
        }
    }

    /// Check a single statement.
    fn check_statement(&mut self, statement: &Statement) -> Result<(), TypeError> {
        match statement {
            Statement::ExpressionStmt(expr) => {
                // Check the expression, ignore its type/value
                self.check_expression(expr)?;
                Ok(())
            }
            Statement::LetBinding {
                name,
                type_ann,
                value,
            } => {
                self.check_binding(name, type_ann, value, false) // is_mutable = false
            }
            Statement::VarBinding {
                name,
                type_ann,
                value,
            } => {
                self.check_binding(name, type_ann, value, true) // is_mutable = true
            }
            Statement::IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                // Check condition is bool
                let cond_type = self.check_expression(condition)?;
                if cond_type != Type::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Bool,
                        found: cond_type,
                        context: "if condition".to_string(),
                    });
                }
                // Check 'then' branch. Append errors if any.
                if let Err(mut branch_errors) = self.check_program(then_branch) {
                    self.errors.append(&mut branch_errors);
                }
                // Check 'else' branch if it exists. Append errors if any.
                if let Some(else_b) = else_branch {
                    if let Err(mut branch_errors) = self.check_program(else_b) {
                        self.errors.append(&mut branch_errors);
                    }
                }
                Ok(())
            }
            Statement::WhileStmt { condition, body } => {
                // Check condition is bool
                let cond_type = self.check_expression(condition)?;
                if cond_type != Type::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Bool,
                        found: cond_type,
                        context: "while condition".to_string(),
                    });
                }
                // Check body. Append errors if any.
                if let Err(mut body_errors) = self.check_program(body) {
                    self.errors.append(&mut body_errors);
                }
                Ok(())
            }
            Statement::ForStmt {
                initializer,
                condition,
                increment,
                body,
            } => {
                // Check initializer (if any)
                if let Some(init) = initializer {
                    if let Err(mut init_errors) = self.check_expression(init) {
                        self.errors.push(init_errors);
                    }
                }
                // Check condition (if any)
                if let Some(cond) = condition {
                    let cond_type = self.check_expression(cond)?;
                    if cond_type != Type::Bool {
                        return Err(TypeError::TypeMismatch {
                            expected: Type::Bool,
                            found: cond_type,
                            context: "for loop condition".to_string(),
                        });
                    }
                }
                // Check increment (if any)
                if let Some(inc) = increment {
                    if let Err(mut inc_errors) = self.check_expression(inc) {
                        self.errors.push(inc_errors);
                    }
                }
                // Check body. Append errors if any.
                if let Err(mut body_errors) = self.check_program(body) {
                    self.errors.append(&mut body_errors);
                }
                Ok(())
            }
            Statement::FunctionDef {
                name,
                params,
                return_type_ann,
                body,
            } => {
                // --- Determine signature ---
                // Resolve Type from annotation or default (needs refinement)
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|(_, opt_ty)| opt_ty.unwrap_or(Type::Float)) // Default Float - BAD! Needs proper checking/inference
                    .collect();
                let return_type = return_type_ann.unwrap_or(Type::Float); // Default Float - BAD!

                let signature = FunctionSignature {
                    param_types: param_types.clone(),
                    return_type,
                };

                // --- Define function ---
                if let Err(e) = self.symbol_table.define_function(name, signature) {
                    return Err(TypeError::FunctionRedefinition(e)); // Use specific error
                }

                // --- Check function body ---
                self.symbol_table.enter_scope(); // Enter function scope
                let original_return_type = self.current_function_return_type.replace(return_type); // Set expected return

                // Define parameters in the new scope
                for (i, (p_name, _)) in params.iter().enumerate() {
                    let param_type = param_types[i];
                    // Parameters are immutable bindings in the function scope
                    let info = SymbolInfo {
                        ty: param_type,
                        is_mutable: false,
                    };
                    if let Err(e) = self.symbol_table.define_variable(p_name, info) {
                        // Add error but might continue checking rest of function
                        self.errors.push(TypeError::VariableRedefinition(e));
                    }
                }

                // Check the body statements
                // Need to track if block actually returns correct type - complex analysis needed.
                // For now, just check statements for errors.
                let _ = self.check_program(body); // Ignore result for now, collect errors in self.errors

                self.current_function_return_type = original_return_type; // Restore outer return type
                self.symbol_table.exit_scope(); // Exit function scope
                Ok(())
            }
        }
    }

    // Helper for LetBinding and VarBinding
    fn check_binding(
        &mut self,
        name: &str,
        type_ann: &Option<Type>,
        value: &Expression,
        is_mutable: bool,
    ) -> Result<(), TypeError> {
        // Check value first (RHS evaluated before variable defined)
        let value_type = self.check_expression(value)?;

        // Determine expected type
        let expected_type = match type_ann {
            Some(ann) => {
                // Type Annotation exists: Check if value type matches annotation
                if value_type != *ann {
                    return Err(TypeError::TypeMismatch {
                        expected: *ann,
                        found: value_type,
                        context: format!("assignment to '{}'", name),
                    });
                }
                *ann // Use the annotation type
            }
            None => value_type, // No annotation: Infer type from value
        };

        // Define variable in current scope
        let info = SymbolInfo {
            ty: expected_type,
            is_mutable,
        };
        self.symbol_table
            .define_variable(name, info)
            .map_err(TypeError::VariableRedefinition)
    }

    /// Check an expression and return its determined Type.
    fn check_expression(&mut self, expression: &Expression) -> Result<Type, TypeError> {
        match expression {
            Expression::FloatLiteral(_) => Ok(Type::Float),
            Expression::IntLiteral(_) => Ok(Type::Int),
            Expression::BoolLiteral(_) => Ok(Type::Bool),
            Expression::Variable(name) => {
                self.symbol_table
                    .lookup_variable(name)
                    .map(|info| info.ty) // Return the found type
                    .ok_or_else(|| TypeError::UndefinedVariable(name.clone()))
            }
            Expression::Assignment { target, value } => {
                // Look up target variable
                let target_info = self
                    .symbol_table
                    .lookup_variable(target)
                    .ok_or_else(|| TypeError::UndefinedVariable(target.clone()))?
                    .clone();
                // Check mutability
                if !target_info.is_mutable {
                    return Err(TypeError::AssignmentToImmutable(target.clone()));
                }
                // Check value type
                let value_type = self.check_expression(value)?;
                // Check type match
                if target_info.ty != value_type {
                    return Err(TypeError::TypeMismatch {
                        expected: target_info.ty,
                        found: value_type,
                        context: format!("assignment to '{}'", target),
                    });
                }
                // Assignment expression type is the type of the assigned value
                Ok(value_type)
            }
            Expression::BinaryOp { op, left, right } => {
                let left_type = self.check_expression(left)?;
                let right_type = self.check_expression(right)?;
                // Basic rules: must match, only int/float allowed for arithmetic
                match (left_type, right_type) {
                    (Type::Int, Type::Int) => Ok(Type::Int), // int + int -> int
                    (Type::Float, Type::Float) => Ok(Type::Float), // float + float -> float
                    _ => Err(TypeError::InvalidOperation {
                        op: format!("{:?}", op), // Improve op formatting later
                        type_info: format!("{} and {}", left_type, right_type),
                    }),
                }
            }
            Expression::ComparisonOp { op, left, right } => {
                let left_type = self.check_expression(left)?;
                let right_type = self.check_expression(right)?;
                // Allow comparing int/int or float/float or bool/bool (for ==, !=)
                if left_type != right_type || left_type == Type::Void {
                    return Err(TypeError::InvalidOperation {
                        op: format!("{:?}", op), // Improve op formatting later
                        type_info: format!("{} and {}", left_type, right_type),
                    });
                }
                if left_type == Type::Bool
                    && !matches!(
                        expression,
                        Expression::ComparisonOp {
                            op: ComparisonOperator::Equal | ComparisonOperator::NotEqual,
                            ..
                        }
                    )
                {
                    return Err(TypeError::InvalidOperation {
                        op: format!("{:?}", op),
                        type_info: "bool and bool".to_string(),
                    });
                }
                // Comparisons always return Bool
                Ok(Type::Bool)
            }
            Expression::UnaryOp { op, operand } => {
                let operand_type = self.check_expression(operand)?;
                match op {
                    UnaryOperator::Negate => {
                        if operand_type == Type::Int || operand_type == Type::Float {
                            Ok(operand_type) // Negation preserves type
                        } else {
                            Err(TypeError::InvalidOperation {
                                op: "-".to_string(),
                                type_info: format!("{}", operand_type),
                            })
                        }
                    }
                    UnaryOperator::Not => {
                        if operand_type == Type::Bool {
                            Ok(Type::Bool) // Not preserves bool
                        } else {
                            Err(TypeError::InvalidOperation {
                                op: "!".to_string(),
                                type_info: format!("{}", operand_type),
                            })
                        }
                    }
                }
            }
            Expression::StringLiteral(string) => {
                Ok(Type::String)
            }
            Expression::FunctionCall { name, args } => {
                // --- Built-in Print ---
                if name == "print" || name == "print_str"{
                    if args.len() != 1 {
                        return Err(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    // Check the type of the argument being printed
                    let _arg_type = self.check_expression(&args[0])?;
                    // TODO: Check if _arg_type is printable (Int, Float, Bool)?
                    Ok(Type::Void) // Make print return Void maybe? Or keep Float(0.0)? Let's try Void.
                }
                // --- User Function Call ---
                else {
                    // Look up function signature
                    let signature = self
                        .symbol_table
                        .lookup_function(name)
                        .ok_or_else(|| TypeError::UndefinedFunction(name.clone()))?
                        .clone();
                    // Check arg count
                    if args.len() != signature.param_types.len() {
                        return Err(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: signature.param_types.len(),
                            found: args.len(),
                        });
                    }
                    // Check arg types
                    for (i, arg_expr) in args.iter().enumerate() {
                        let arg_type = self.check_expression(arg_expr)?;
                        let expected_type = signature.param_types[i];
                        if arg_type != expected_type {
                            return Err(TypeError::TypeMismatch {
                                expected: expected_type,
                                found: arg_type,
                                context: format!("argument {} of function '{}'", i, name),
                            });
                        }
                    }
                    // Return the function's declared return type
                    Ok(signature.return_type)
                }
            }
            Expression::IfExpr {
                condition,
                then_branch,
                else_branch,
            } => {
                // Check condition is bool
                let cond_type = self.check_expression(condition)?;
                if cond_type != Type::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: Type::Bool,
                        found: cond_type,
                        context: "if condition".to_string(),
                    });
                }
                // Check branches and ensure they have the same type
                let then_type = self.check_expression(then_branch)?;
                let else_type = self.check_expression(else_branch)?;
                if then_type == Type::Void || else_type == Type::Void || then_type != else_type {
                    return Err(TypeError::TypeMismatch {
                        expected: then_type,
                        found: else_type,
                        context: "if expression branches".to_string(),
                    });
                }
                // Result type is the type of the branches
                Ok(then_type)
            }
            Expression::Block {
                statements,
                final_expression,
            } => {
                // Check statements in a new scope? No, let block inherit scope for now.
                self.symbol_table.enter_scope(); // Check block in its own variable scope
                for stmt in statements {
                    self.check_statement(stmt)?;
                }
                // Determine result type from final expression or default?
                let block_type = if let Some(final_expr) = final_expression {
                    self.check_expression(final_expr)?
                } else {
                    Type::Void // Block with no final expression has Void type
                };
                self.symbol_table.exit_scope();
                Ok(block_type)
            }
        }
    }
} // End impl TypeChecker

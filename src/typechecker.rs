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
    IfBranchMismatch {
        // Specific error for IfExpr
        then_type: Type,
        else_type: Type,
    },
    InvalidConditionType(Type),      // For if/while condition
    InvalidAssignmentTarget(String), // If LHS of = isn't assignable
    VoidAssignment(String),          // Assigning void to variable
    PrintArgError(String),           // Error for built-in print type
    UnknownTypeName(String),         // Type name in annotation not recognized
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
            TypeError::IfBranchMismatch {
                then_type,
                else_type,
            } => write!(
                f,
                "If expression branches have different types: {} vs {}",
                then_type, else_type
            ),
            TypeError::InvalidConditionType(found) => {
                write!(f, "Condition must be boolean, found {}", found)
            }
            TypeError::InvalidAssignmentTarget(target) => {
                write!(f, "Invalid target for assignment: {}", target)
            }
            TypeError::VoidAssignment(name) => {
                write!(f, "Cannot assign void value to variable '{}'", name)
            }
            TypeError::PrintArgError(msg) => write!(f, "Built-in print error: {}", msg),
            TypeError::UnknownTypeName(name) => write!(f, "Unknown type name '{}'", name),
        }
    }
}

pub struct TypeChecker {
    symbol_table: SymbolTable,
    errors: Vec<TypeError>,
    // Track expected return type for function bodies
    current_function_return_type: Option<Type>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
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
        for statement in &program.statements {
            if let Statement::FunctionDef {
                name,
                params,
                return_type_ann,
                ..
            } = statement
            {
                // Basic type resolution (replace with better handling)
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|(_, opt_ty)| opt_ty.unwrap_or(Type::Float)) // Default BAD
                    .collect();
                let return_type = return_type_ann.unwrap_or(Type::Float); // Default BAD
                let signature = FunctionSignature {
                    param_types,
                    return_type,
                };
                if let Err(e) = self.symbol_table.define_function(name, signature) {
                    self.errors.push(TypeError::FunctionRedefinition(e));
                }
            }
        }

        // --- Second Pass: Check Statements ---
        for statement in &program.statements {
            self.check_statement(statement); // Collect errors in self.errors
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    /// Check a single statement.
    /// Check a single statement, adding errors to self.errors.
    fn check_statement(&mut self, statement: &Statement) {
        // No longer returns Result
        match statement {
            Statement::ExpressionStmt(expr) => {
                // Check expression for side effects and errors, ignore resulting type
                let _ = self.check_expression(expr);
            }
            Statement::LetBinding {
                name,
                type_ann,
                value,
            } => {
                self.check_binding(name, type_ann, value, false); // is_mutable = false
            }
            Statement::VarBinding {
                name,
                type_ann,
                value,
            } => {
                self.check_binding(name, type_ann, value, true); // is_mutable = true
            }
            Statement::IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                // Check condition is bool
                match self.check_expression(condition) {
                    Some(Type::Bool) => {} // OK
                    Some(other_type) => self
                        .errors
                        .push(TypeError::InvalidConditionType(other_type)),
                    None => {} // Error already recorded
                }
                // Check branches (maybe enter scope?)
                self.symbol_table.enter_scope(); // Check branches in new scope
                let _ = self.check_program_block(then_branch);
                self.symbol_table.exit_scope();

                if let Some(else_b) = else_branch {
                    self.symbol_table.enter_scope();
                    let _ = self.check_program_block(else_b);
                    self.symbol_table.exit_scope();
                }
            }
            Statement::WhileStmt { condition, body } => {
                // Check condition is bool
                match self.check_expression(condition) {
                    Some(Type::Bool) => {}
                    Some(other_type) => self
                        .errors
                        .push(TypeError::InvalidConditionType(other_type)),
                    None => {}
                }
                // Check body
                self.symbol_table.enter_scope(); // Check body in new scope
                let _ = self.check_program_block(body);
                self.symbol_table.exit_scope();
            }
            Statement::ForStmt {
                initializer,
                condition,
                increment,
                body,
            } => {
                // Check for needs proper scoping if init declares vars
                self.symbol_table.enter_scope(); // Scope for initializer and loop
                if let Some(init_expr) = initializer {
                    self.check_expression(init_expr);
                }
                if let Some(cond_expr) = condition {
                    match self.check_expression(cond_expr) {
                        Some(Type::Bool) | None => {} // Allow None if error during check
                        Some(other) => self.errors.push(TypeError::InvalidConditionType(other)),
                    }
                }
                // Check increment *after* body check? Doesn't matter much here.
                if let Some(incr_expr) = increment {
                    self.check_expression(incr_expr);
                }
                // Check body
                let _ = self.check_program_block(body);
                self.symbol_table.exit_scope();
            }
            Statement::FunctionDef {
                name,
                params,
                return_type_ann,
                body,
            } => {
                // Signature already collected in first pass. Now check the body.
                let signature = self.symbol_table.lookup_function(name).cloned(); // Clone to satisfy borrow checker
                if signature.is_none() {
                    return;
                } // Skip body check if definition failed
                let signature = signature.unwrap();

                self.symbol_table.enter_scope(); // Enter function scope
                let original_return_type = self
                    .current_function_return_type
                    .replace(signature.return_type);

                // Define parameters in the new scope
                for (i, (p_name, _)) in params.iter().enumerate() {
                    if i < signature.param_types.len() {
                        // Check bounds
                        let param_type = signature.param_types[i];
                        let info = SymbolInfo {
                            ty: param_type,
                            is_mutable: false,
                        }; // Params are immutable
                        if let Err(e) = self.symbol_table.define_variable(p_name, info) {
                            self.errors.push(TypeError::VariableRedefinition(e));
                        }
                    }
                }

                // Check the body statements
                let _ = self.check_program_block(body); // Check body, collect errors

                // TODO: Check return value consistency within body - complex!

                self.current_function_return_type = original_return_type;
                self.symbol_table.exit_scope();
            }
        }
    }

    // Helper for LetBinding and VarBinding
    // Helper for LetBinding and VarBinding - pushes errors
    fn check_binding(
        &mut self,
        name: &str,
        type_ann: &Option<Type>,
        value: &Expression,
        is_mutable: bool,
    ) {
        let Some(value_type) = self.check_expression(value) else {
            return;
        }; // Check value first

        // Determine expected type
        let expected_type = match type_ann {
            Some(ann) => {
                if value_type == Type::Void {
                    // Cannot assign void
                    self.errors
                        .push(TypeError::VoidAssignment(name.to_string()));
                    return; // Stop checking this binding
                }
                if value_type != *ann {
                    self.errors.push(TypeError::TypeMismatch {
                        expected: *ann,
                        found: value_type,
                        context: format!("binding of '{}'", name),
                    });
                    // Use annotation type even if mismatch, allows defining variable
                }
                *ann
            }
            None => {
                // Infer from value
                if value_type == Type::Void {
                    // Cannot infer variable type from void
                    self.errors
                        .push(TypeError::VoidAssignment(name.to_string()));
                    return;
                }
                value_type
            }
        };

        // Define variable in current scope
        let info = SymbolInfo {
            ty: expected_type,
            is_mutable,
        };
        if let Err(e) = self.symbol_table.define_variable(name, info) {
            self.errors.push(TypeError::VariableRedefinition(e));
        }
    }

    /// Check an expression and return its determined Type or None if error.
    /// Errors are pushed to self.errors.
    fn check_expression(&mut self, expression: &Expression) -> Option<Type> {
        match expression {
            // ... Literals, Variable (use symbol_table.lookup_variable) ...
            Expression::FloatLiteral(_) => Some(Type::Float),
            Expression::IntLiteral(_) => Some(Type::Int),
            Expression::BoolLiteral(_) => Some(Type::Bool),
            Expression::Variable(name) => {
                self.symbol_table
                    .lookup_variable(name)
                    .map(|info| info.ty) // Return the found type
                    .or_else(|| {
                        self.errors.push(TypeError::UndefinedVariable(name.clone()));
                        None
                    })
            }
            Expression::Assignment { target, value } => {
                // Check target exists and is mutable
                let target_info = match self.symbol_table.lookup_variable(target) {
                    Some(info) => *info, // Copy info
                    None => {
                        self.errors
                            .push(TypeError::UndefinedVariable(target.clone()));
                        return None;
                    }
                };
                if !target_info.is_mutable {
                    self.errors
                        .push(TypeError::AssignmentToImmutable(target.clone()));
                    // Continue checking value type, but assignment itself is invalid type-wise?
                    // Let's return None to signify the assignment expression itself has an error
                    let _ = self.check_expression(value); // Still check RHS for errors
                    return None;
                }

                // Check value type matches target type
                let Some(value_type) = self.check_expression(value) else {
                    return None;
                };
                if value_type == Type::Void {
                    self.errors.push(TypeError::VoidAssignment(target.clone()));
                    return None;
                }
                if target_info.ty != value_type {
                    self.errors.push(TypeError::TypeMismatch {
                        expected: target_info.ty,
                        found: value_type,
                        context: format!("assignment to '{}'", target),
                    });
                    return None; // Assignment fails type check
                }
                Some(value_type) // Assignment yields value type
            }
            Expression::BinaryOp { op, left, right } => {
                let left_type_opt = self.check_expression(left);
                let right_type_opt = self.check_expression(right);
                match (left_type_opt, right_type_opt) {
                    (Some(Type::Int), Some(Type::Int)) => Some(Type::Int),
                    (Some(Type::Float), Some(Type::Float)) => Some(Type::Float),
                    (Some(lt), Some(rt)) => {
                        // Mismatch or non-numeric
                        self.errors.push(TypeError::InvalidOperation {
                            op: format!("{:?}", op),
                            type_info: format!("{} and {}", lt, rt),
                        });
                        None
                    }
                    _ => None, // Error in operand(s)
                }
            }
            Expression::ComparisonOp { op, left, right } => {
                let left_type_opt = self.check_expression(left);
                let right_type_opt = self.check_expression(right);
                match (left_type_opt, right_type_opt) {
                    (Some(lt @ Type::Int), Some(Type::Int)) => Some(Type::Bool),
                    (Some(lt @ Type::Float), Some(Type::Float)) => Some(Type::Bool),
                    (Some(lt @ Type::Bool), Some(Type::Bool))
                        if matches!(
                            op,
                            ComparisonOperator::Equal | ComparisonOperator::NotEqual
                        ) =>
                    {
                        Some(Type::Bool)
                    }
                    (Some(lt), Some(rt)) => {
                        // Mismatch or invalid op
                        self.errors.push(TypeError::InvalidOperation {
                            op: format!("{:?}", op),
                            type_info: format!("{} and {}", lt, rt),
                        });
                        None
                    }
                    _ => None, // Error in operand(s)
                }
            }
            Expression::UnaryOp { op, operand } => {
                let operand_type_opt = self.check_expression(operand);
                match (op, operand_type_opt) {
                    (UnaryOperator::Negate, Some(Type::Int)) => Some(Type::Int),
                    (UnaryOperator::Negate, Some(Type::Float)) => Some(Type::Float),
                    (UnaryOperator::Not, Some(Type::Bool)) => Some(Type::Bool),
                    (UnaryOperator::Negate, Some(other)) => {
                        self.errors.push(TypeError::InvalidOperation {
                            op: format!("{:?}", op),
                            type_info: format!("{}", other),
                        });
                        None
                    }
                    _ => None, // Error in operand
                }
            }
            Expression::StringLiteral(string) => Some(Type::String),
            Expression::FunctionCall { name, args } => {
                // --- Built-ins ---
                if name == "print"
                    || name == "print_str"
                    || name == "print_int"
                    || name == "print_bool"
                {
                    if args.len() != 1 {
                        /* Error: incorrect arg count */
                        self.errors.push(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    } else {
                        let _ = self.check_expression(&args[0]); /* Check arg exists/is valid */
                    }
                    // TODO: More specific type check for each print variant if needed
                    Some(Type::Void) // Print calls evaluate to Void
                }
                // --- User Functions ---
                else {
                    match self.symbol_table.lookup_function(name).cloned() {
                        // Clone sig
                        Some(signature) => {
                            if args.len() != signature.param_types.len() { /* Error */ }
                            // Check arg types
                            for i in 0..std::cmp::min(args.len(), signature.param_types.len()) {
                                let Some(arg_type) = self.check_expression(&args[i]) else {
                                    continue;
                                };
                                let expected = signature.param_types[i];
                                if arg_type != expected { /* Error */ }
                            }
                            Some(signature.return_type)
                        }
                        None => {
                            self.errors.push(TypeError::UndefinedFunction(name.clone()));
                            None
                        }
                    }
                }
            }
            Expression::IfExpr {
                condition,
                then_branch,
                else_branch,
            } => {
                match self.check_expression(condition) {
                    Some(Type::Bool) => {} // OK
                    Some(other) => {
                        self.errors.push(TypeError::InvalidConditionType(other));
                        /* Fallthrough to check branches */
                    }
                    None => {} // Error checking condition already recorded
                }
                let then_type_opt = self.check_expression(then_branch);
                let else_type_opt = self.check_expression(else_branch);
                match (then_type_opt, else_type_opt) {
                    (Some(tt), Some(et)) => {
                        if tt == Type::Void || tt != et {
                            // Branches cannot be void and must match
                            self.errors.push(TypeError::IfBranchMismatch {
                                then_type: tt,
                                else_type: et,
                            });
                            None // Type mismatch means expression has no valid type
                        } else {
                            Some(tt)
                        }
                    }
                    _ => None, // Error occurred in one or both branches
                }
            }
            Expression::Block {
                statements,
                final_expression,
            } => {
                self.symbol_table.enter_scope();
                for stmt in statements {
                    self.check_statement(stmt);
                }
                let block_type = if let Some(final_expr) = final_expression {
                    self.check_expression(final_expr) // Type is final expression's type (or None if error)
                } else {
                    Some(Type::Void)
                }; // No final expr -> Void
                self.symbol_table.exit_scope();
                block_type
            }
            _ => unimplemented!(
                "Type checking not implemented for this expression node yet: {:?}",
                expression
            ),
        }
    } // End check_expression

    // Helper to check blocks used inside statements like IfStmt, WhileStmt, ForStmt
    // These blocks don't produce a value for the statement itself.
    fn check_program_block(&mut self, program: &Program) {
        // Simply check statements, collecting errors in self.errors
        for stmt in &program.statements {
            self.check_statement(stmt);
        }
    }
} // End impl TypeChecker

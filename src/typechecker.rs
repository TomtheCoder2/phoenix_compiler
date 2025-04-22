// src/typechecker.rs

use crate::ast::{ComparisonOperator, Expression, ExpressionKind, Program, Statement, StatementKind, UnaryOperator};
use crate::location::{Location, Span};
use crate::symbol_table::{FunctionSignature, SymbolInfo, SymbolTable};
use crate::types::Type;
use std::fmt;
use std::fmt::write;

#[derive(Debug, Clone, PartialEq)]
pub enum TypeError {
    // Add Span to relevant errors
    UndefinedVariable(String, Span),
    UndefinedFunction(String, Span),
    VariableRedefinition {
        name: String,
        new_loc: Span,
        prev_loc: Span,
    }, // Need prev location from symbol table
    FunctionRedefinition {
        name: String,
        new_loc: Span,
        prev_loc: Span,
    },
    AssignmentToImmutable(String, Span),
    TypeMismatch {
        expected: Type,
        found: Type,
        context: String,
        span: Span,
    },
    IncorrectArgCount {
        func_name: String,
        expected: usize,
        found: usize,
        call_site: Span,
    },
    InvalidOperation {
        op: String,
        type_info: String,
        span: Span,
    },
    MissingReturnValue(String, Span), // Span of function definition?
    IfBranchMismatch {
        then_type: Type,
        else_type: Type,
        span: Span,
    }, // Span of IfExpr
    InvalidConditionType(Type, Span), // Span of condition expr
    InvalidAssignmentTarget(String, Span),
    VoidAssignment(String, Span), // Span of assignment or variable binding
    PrintArgError(String, Span),  // Span of print call argument
    UnknownTypeName(String, Location), // Location of type name identifier
    ReturnVoidFromNonVoid(Type, Span), // Span of return statement
    ReturnValueFromVoid(Span),
    ReturnTypeMismatch { expected: Type, found: Type, span: Span }, // Span of return statement
}
// src/typechecker.rs
impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeError::UndefinedVariable(n, span) => {
                write!(f, "{} Undefined variable '{}'", span, n)
            }
            TypeError::TypeMismatch {
                span,
                expected,
                found,
                context,
            } => write!(
                f,
                "{} Type mismatch in {}: expected {}, found {}",
                span, context, expected, found
            ),
            TypeError::VariableRedefinition { name, new_loc, prev_loc } => {
                write!(f, "{} Variable '{}' redefined in this scope, first definition at {}", new_loc, name, prev_loc)
            }
            TypeError::FunctionRedefinition { name, new_loc, .. } => {
                write!(f, "{} Function '{}' redefined", new_loc, name)
            }
            TypeError::AssignmentToImmutable(n, span) => {
                write!(f, "{} Cannot assign to immutable variable '{}'", span, n)
            }
            TypeError::IncorrectArgCount {
                func_name,
                expected,
                found,
                call_site,
            } => write!(
                f,
                "{} Function '{}' called with {} arguments, expected {}",
                call_site, func_name, found, expected
            ),
            TypeError::InvalidOperation { op, type_info, span } => {
                write!(f, "{} Invalid operation '{}' for type(s) {}", span, op, type_info)
            }
            TypeError::MissingReturnValue(fname, span) => write!(
                f,
                "{} Function '{}' may not return a value in all paths",
                span, fname
            ),
            TypeError::IfBranchMismatch {
                then_type,
                else_type,
                span,
            } => write!(
                f,
                "{} If expression branches have different types: {} vs {}",
                span, then_type, else_type
            ),
            TypeError::InvalidConditionType(found, span) => {
                write!(f, "{} Condition must be boolean, found {}", span, found)
            }
            TypeError::InvalidAssignmentTarget(target, span) => {
                write!(f, "{} Invalid target for assignment: {}", span, target)
            }
            TypeError::VoidAssignment(name, span) => {
                write!(f, "{} Cannot assign void value to variable '{}'", span, name)
            }
            TypeError::PrintArgError(msg, span) => write!(f, "{} Built-in print error: {}", span, msg),
            TypeError::UnknownTypeName(name, loc) => write!(f, "{} Unknown type name '{}'", loc, name),
            TypeError::UndefinedFunction(name, span) => {
                write!(f, "{} Undefined function '{}'", span, name)
            }
            TypeError::ReturnVoidFromNonVoid(expected_type, span) => {
                write!(f, "{} Cannot return void from non-void function, expected {}", span, expected_type)
            },
            TypeError::ReturnValueFromVoid(span)  => {
                write!(f, "{} Cannot return a value from a void function", span)
            }
            TypeError::ReturnTypeMismatch { expected, found, span } => {
                write!(f, "{} Return type mismatch: expected {}, found {}", span, expected, found)
            }
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
            let span = statement.span.clone();
            if let StatementKind::FunctionDef {
                name,
                params,
                return_type_ann,
                ..
            } = &statement.kind
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
                    def_span: span.clone(),
                };
                if let Err(e) = self.symbol_table.define_function(name, signature) {
                    // e holds the span where the function was defined
                    self.errors.push(TypeError::FunctionRedefinition {
                        name: name.clone(),
                        new_loc: span,
                        prev_loc: e,
                    });
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
        let span = statement.span.clone();
        // No longer returns Result
        match &statement.kind {
            StatementKind::ExpressionStmt(expr) => {
                // Check expression for side effects and errors, ignore resulting type
                let _ = self.check_expression(expr);
            }
            StatementKind::LetBinding {
                name,
                type_ann,
                value,
            } => {
                self.check_binding(name, type_ann, value, false); // is_mutable = false
            }
            StatementKind::VarBinding {
                name,
                type_ann,
                value,
            } => {
                self.check_binding(name, type_ann, value, true); // is_mutable = true
            }
            StatementKind::IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                // Check condition is bool
                match self.check_expression(condition) {
                    Some(Type::Bool) => {} // OK
                    Some(other_type) => self
                        .errors
                        .push(TypeError::InvalidConditionType(other_type, span)),
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
            StatementKind::WhileStmt { condition, body } => {
                // Check condition is bool
                match self.check_expression(condition) {
                    Some(Type::Bool) => {}
                    Some(other_type) => self
                        .errors
                        .push(TypeError::InvalidConditionType(other_type, span)),
                    None => {}
                }
                // Check body
                self.symbol_table.enter_scope(); // Check body in new scope
                let _ = self.check_program_block(body);
                self.symbol_table.exit_scope();
            }
            StatementKind::ForStmt {
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
                        Some(other) => self.errors.push(TypeError::InvalidConditionType(other, cond_expr.span.clone())),
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
            StatementKind::FunctionDef {
                name,
                params,
                return_type_ann,
                body,
            } => {
                // Signature already collected in first pass. Now check the body.
                let signature  = self.symbol_table.lookup_function(name).cloned(); // Clone to satisfy borrow checker
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
                            def_span: signature.def_span.clone(),
                        }; // Params are immutable
                        if let Err(e) = self.symbol_table.define_variable(p_name, info) {
                            self.errors.push(TypeError::VariableRedefinition {
                                name: p_name.clone(),
                                new_loc: signature.def_span.clone(),
                                prev_loc: e,
                            });
                        }
                    }
                }

                // Check the body statements
                self.check_program_block(body); // Check body, collect errors

                // TODO: Check return value consistency within body - complex!

                self.current_function_return_type = original_return_type;
                self.symbol_table.exit_scope();
            }
            StatementKind::ReturnStmt { value } => {
                // Check context: Must be inside a function
                let Some(expected_ret_type) = self.current_function_return_type else {
                    self.errors.push(TypeError::InvalidOperation {
                        op: "return".to_string(),
                        type_info: "Cannot return from top-level code".to_string(),
                        span,
                    });
                    return; // Stop checking this statement
                };

                match (value, expected_ret_type) {
                    // Case 1: return;
                    (None, Type::Void) => { /* OK: Returning void from void function */ }
                    (None, non_void_type) => {
                        // Error: return; from function expecting a value
                        self.errors
                            .push(TypeError::ReturnVoidFromNonVoid(non_void_type, span));
                    }
                    // Case 2: return <expr>;
                    (Some(expr), Type::Void) => {
                        // Error: Returning a value from void function
                        // Still check the expression itself for errors though
                        let _ = self.check_expression(expr);
                        self.errors.push(TypeError::ReturnValueFromVoid(span));
                    }
                    (Some(expr), expected_type) => {
                        // Check the expression and its type
                        if let Some(found_type) = self.check_expression(expr) {
                            if found_type == Type::Void {
                                // Cannot return result of a void expression (e.g. return print(x);)
                                self.errors.push(TypeError::ReturnTypeMismatch {
                                    expected: expected_type,
                                    found: Type::Void,
                                    span: expr.span.clone(),
                                });
                            } else if found_type != expected_type {
                                self.errors.push(TypeError::ReturnTypeMismatch {
                                    expected: expected_type,
                                    found: found_type,
                                    span: expr.span.clone(),
                                });
                            }
                        }
                        // If check_expression returned None, error already recorded
                    }
                }
            } // End ReturnStmt
        }
    }

    // Helper for LetBinding and VarBinding - pushes errors
    // todo: currently the value is used for the def_span. make it such that the whole stmt is used for the def_span. because 
    // then the errors get more usable imo
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
                        .push(TypeError::VoidAssignment(name.to_string(), value.span.clone()));
                    return; // Stop checking this binding
                }
                if value_type != *ann {
                    self.errors.push(TypeError::TypeMismatch {
                        expected: *ann,
                        found: value_type,
                        context: format!("binding of '{}'", name),
                        span: value.span.clone(),
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
                        .push(TypeError::VoidAssignment(name.to_string(), value.span.clone()));
                    return;
                }
                value_type
            }
        };

        // Define variable in current scope
        let info = SymbolInfo {
            ty: expected_type,
            is_mutable,
            def_span: value.span.clone(),
        };
        if let Err(e) = self.symbol_table.define_variable(name, info) {
            self.errors.push(TypeError::VariableRedefinition {
                name: name.to_string(),
                new_loc: value.span.clone(),
                prev_loc: e,
            });
        }
    }

    /// Check an expression and return its determined Type or None if error.
    /// Errors are pushed to self.errors.
    fn check_expression(&mut self, expression: &Expression) -> Option<Type> {
        let span = expression.span.clone();
        match &expression.kind {
            // ... Literals, Variable (use symbol_table.lookup_variable) ...
            ExpressionKind::FloatLiteral(_) => Some(Type::Float),
            ExpressionKind::IntLiteral(_) => Some(Type::Int),
            ExpressionKind::BoolLiteral(_) => Some(Type::Bool),
            ExpressionKind::Variable(name) => {
                self.symbol_table
                    .lookup_variable(name)
                    .map(|info| info.ty) // Return the found type
                    .or_else(|| {
                        self.errors.push(TypeError::UndefinedVariable(name.clone(), span));
                        None
                    })
            }
            ExpressionKind::Assignment { target, value } => {
                // Check target exists and is mutable
                let target_info = match self.symbol_table.lookup_variable(target) {
                    Some(info) => info, // Copy info
                    None => {
                        self.errors
                            .push(TypeError::UndefinedVariable(target.clone(), span));
                        return None;
                    }
                }.clone();
                if !target_info.is_mutable {
                    self.errors
                        .push(TypeError::AssignmentToImmutable(target.clone(), span));
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
                    self.errors.push(TypeError::VoidAssignment(target.clone(), span));
                    return None;
                }
                if target_info.ty != value_type {
                    self.errors.push(TypeError::TypeMismatch {
                        expected: target_info.ty,
                        found: value_type,
                        context: format!("assignment to '{}'", target),
                        span
                    });
                    return None; // Assignment fails type check
                }
                Some(value_type) // Assignment yields value type
            }
            ExpressionKind::BinaryOp { op, left, right } => {
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
                            span
                        });
                        None
                    }
                    _ => None, // Error in operand(s)
                }
            }
            ExpressionKind::ComparisonOp { op, left, right } => {
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
                            span
                        });
                        None
                    }
                    _ => None, // Error in operand(s)
                }
            }
            ExpressionKind::UnaryOp { op, operand } => {
                let operand_type_opt = self.check_expression(operand);
                match (op, operand_type_opt) {
                    (UnaryOperator::Negate, Some(Type::Int)) => Some(Type::Int),
                    (UnaryOperator::Negate, Some(Type::Float)) => Some(Type::Float),
                    (UnaryOperator::Not, Some(Type::Bool)) => Some(Type::Bool),
                    (UnaryOperator::Negate, Some(other)) => {
                        self.errors.push(TypeError::InvalidOperation {
                            op: format!("{:?}", op),
                            type_info: format!("{}", other),
                            span
                        });
                        None
                    }
                    _ => None, // Error in operand
                }
            }
            ExpressionKind::StringLiteral(string) => Some(Type::String),
            ExpressionKind::FunctionCall { name, args } => {
                // --- Built-ins ---
                if name == "print"
                    || name == "print_str"
                    || name == "print_int"
                    || name == "print_bool"
                    || name == "print_float"
                    || name == "println"
                {
                    if args.len() != 1 {
                        /* Error: incorrect arg count */
                        self.errors.push(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: 1,
                            found: args.len(),
                            call_site: span,
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
                            self.errors.push(TypeError::UndefinedFunction(name.clone(), span));
                            None
                        }
                    }
                }
            }
            ExpressionKind::IfExpr {
                condition,
                then_branch,
                else_branch,
            } => {
                match self.check_expression(condition) {
                    Some(Type::Bool) => {} // OK
                    Some(other) => {
                        self.errors.push(TypeError::InvalidConditionType(other, span.clone()));
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
                                span: span.clone(),
                            });
                            None // Type mismatch means expression has no valid type
                        } else {
                            Some(tt)
                        }
                    }
                    _ => None, // Error occurred in one or both branches
                }
            }
            ExpressionKind::Block {
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

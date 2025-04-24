// src/typechecker.rs

use crate::ast::{
    BinaryOperator, ComparisonOperator, Expression, ExpressionKind, LogicalOperator, Program,
    Statement, StatementKind, TypeNode, TypeNodeKind, UnaryOperator,
};
use crate::location::{Location, Span};
use crate::symbol_table::{FunctionSignature, SymbolInfo, SymbolTable};
use crate::types::Type;
use std::fmt;

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
    ReturnTypeMismatch {
        expected: Type,
        found: Type,
        span: Span,
    }, // Span of return statement
    NotAVector(Type, Span),      // Tried to index non-vector
    IndexNotInteger(Type, Span), // Index expr not int
    VectorElementTypeMismatch {
        expected: Type,
        found: Type,
        span: Span,
    }, // In literal or assignment
    CannotInferVectorType(Span), // e.g., for empty literal []
    PushNotVector(Type, Span),   // Target of push isn't vector
    PushElementTypeMismatch {
        vec_type: Type,
        elem_type: Type,
        span: Span,
    }, // Wrong type pushed
    AssignIndexTargetNotVector(Type, Span), // Target of index assign isn't vector
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
            TypeError::VariableRedefinition {
                name,
                new_loc,
                prev_loc,
            } => {
                write!(
                    f,
                    "{} Variable '{}' redefined in this scope, first definition at {}",
                    new_loc, name, prev_loc
                )
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
            TypeError::InvalidOperation {
                op,
                type_info,
                span,
            } => {
                write!(
                    f,
                    "{} Invalid operation '{}' for type(s) {}",
                    span, op, type_info
                )
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
                write!(
                    f,
                    "{} Cannot assign void value to variable '{}'",
                    span, name
                )
            }
            TypeError::PrintArgError(msg, span) => {
                write!(f, "{} Built-in print error: {}", span, msg)
            }
            TypeError::UnknownTypeName(name, loc) => {
                write!(f, "{} Unknown type name '{}'", loc, name)
            }
            TypeError::UndefinedFunction(name, span) => {
                write!(f, "{} Undefined function '{}'", span, name)
            }
            TypeError::ReturnVoidFromNonVoid(expected_type, span) => {
                write!(
                    f,
                    "{} Cannot return void from non-void function, expected {}",
                    span, expected_type
                )
            }
            TypeError::ReturnValueFromVoid(span) => {
                write!(f, "{} Cannot return a value from a void function", span)
            }
            TypeError::ReturnTypeMismatch {
                expected,
                found,
                span,
            } => {
                write!(
                    f,
                    "{} Return type mismatch: expected {}, found {}",
                    span, expected, found
                )
            }
            TypeError::NotAVector(ty, span) => {
                write!(f, "{} Not a vector type: {}", span, ty)
            }
            TypeError::IndexNotInteger(ty, span) => {
                write!(f, "{} Index must be an integer, found {}", span, ty)
            }
            TypeError::VectorElementTypeMismatch {
                expected,
                found,
                span,
            } => {
                write!(
                    f,
                    "{} Vector element type mismatch: expected {}, found {}",
                    span, expected, found
                )
            }
            TypeError::CannotInferVectorType(span) => {
                write!(f, "{} Cannot infer vector type from empty literal", span)
            }
            TypeError::PushNotVector(ty, span) => {
                write!(f, "{} Cannot push to type {} (must be a vector)", span, ty)
            }
            TypeError::PushElementTypeMismatch {
                vec_type,
                elem_type,
                span,
            } => {
                // Wrong type pushed
                write!(f, "{} Cannot push element with type {} to vector containing elements with type {}", span, elem_type, vec_type)
            }
            TypeError::AssignIndexTargetNotVector(ty, span) => {
                write!(f, "{} Cannot index into type {}", span, ty)
            }
        }
    }
}

// Helper to resolve Type AST to internal Type enum
fn resolve_type_node(node: &TypeNode, errors: &mut Vec<TypeError>) -> Option<Type> {
    match &node.kind {
        TypeNodeKind::Simple(name) => {
            match name.as_str() {
                "int" => Some(Type::Int),
                "float" => Some(Type::Float),
                "bool" => Some(Type::Bool),
                "string" => Some(Type::String), // Handle string type name
                "void" => Some(Type::Void),
                _ => {
                    errors.push(TypeError::UnknownTypeName(
                        name.clone(),
                        node.span.start.clone(),
                    ));
                    None
                }
            }
        }
        TypeNodeKind::Vector(element_node) => {
            // Recursively resolve element type
            resolve_type_node(element_node, errors).map(|t| Type::Vector(Box::new(t)))
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
                // type resolution
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|(_, opt_node)| {
                        opt_node
                            .as_ref()
                            .and_then(|n| resolve_type_node(n, &mut self.errors))
                    })
                    .collect::<Option<Vec<Type>>>() // Collects Option<Type>, fails if any param type is None
                    .unwrap_or_else(|| vec![]); // Default empty vector if resolution failed
                let return_type = return_type_ann
                    .as_ref()
                    .and_then(|n| resolve_type_node(n, &mut self.errors))
                    .unwrap_or(Type::Void); // Default to Void if resolution failed
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
                let _ = self.check_expression(expr, None);
            }
            StatementKind::LetBinding {
                name,
                type_ann,
                value,
            } => {
                let annotation_type: Option<Type> = type_ann
                    .as_ref()
                    .and_then(|node| self.resolve_type_node(node));
                self.check_binding(name, &annotation_type, value, false, span); // is_mutable = false
            }
            StatementKind::VarBinding {
                name,
                type_ann,
                value,
            } => {
                let annotation_type: Option<Type> = type_ann
                    .as_ref()
                    .and_then(|node| self.resolve_type_node(node));
                self.check_binding(name, &annotation_type, value, true, span); // is_mutable = true
            }
            StatementKind::IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                // Check condition is bool
                match self.check_expression(condition, None) {
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
                match self.check_expression(condition, None) {
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
                    self.check_statement(init_expr);
                }
                if let Some(cond_expr) = condition {
                    match self.check_expression(cond_expr, None) {
                        Some(Type::Bool) | None => {} // Allow None if error during check
                        Some(other) => self.errors.push(TypeError::InvalidConditionType(
                            other,
                            cond_expr.span.clone(),
                        )),
                    }
                }
                // Check increment *after* body check? Doesn't matter much here.
                if let Some(incr_expr) = increment {
                    self.check_expression(incr_expr, None);
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
                let signature = self.symbol_table.lookup_function(name).cloned(); // Clone to satisfy borrow checker
                if signature.is_none() {
                    return;
                } // Skip body check if definition failed
                let signature = signature.unwrap();

                self.symbol_table.enter_scope(); // Enter function scope
                self.current_function_return_type
                    .replace(signature.return_type);

                // Resolve param/return types from TypeNodes
                let param_types_result: Option<Vec<Type>> = params
                    .iter()
                    .map(|(_, opt_node)| {
                        opt_node
                            .as_ref()
                            .and_then(|n| resolve_type_node(n, &mut self.errors))
                    })
                    .collect(); // Collects Option<Type>, fails if any param type is None
                let return_type_result: Option<Type> = return_type_ann
                    .as_ref()
                    .and_then(|n| resolve_type_node(n, &mut self.errors));

                // Define parameters in the new scope
                for (i, (p_name, _)) in params.iter().enumerate() {
                    if i < signature.param_types.len() {
                        // Check bounds
                        let param_type = signature.param_types[i].clone();
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

                // Use defaults only if resolution failed AND annotation was None? Needs thought.
                // For now, proceed only if all types resolved.
                if let (Some(param_types), Some(return_type)) =
                    (param_types_result, return_type_result.or(Some(Type::Void)))
                {
                    // Default return void if None? Or error? Default Void.
                    let signature = FunctionSignature {
                        param_types,
                        return_type,
                        def_span: statement.span.clone(),
                    };
                    if self.symbol_table.lookup_function(name).is_none() {
                        // Define only if not already defined (first pass might have failed)
                        if let Err(e) = self.symbol_table.define_function(name, signature.clone()) {
                            self.errors.push(TypeError::FunctionRedefinition {
                                name: name.clone(),
                                new_loc: statement.span.clone(),
                                prev_loc: e,
                            });
                        }
                    }
                    // Check body... (pass resolved signature down if needed)
                    self.symbol_table.enter_scope();
                    self.current_function_return_type = Some(signature.return_type);
                    // Define params with resolved types...
                    self.check_program_block(body);
                    self.symbol_table.exit_scope();
                    self.current_function_return_type = None; // Restore
                }
            }
            StatementKind::ReturnStmt { value } => {
                // Check context: Must be inside a function
                let Some(expected_ret_type) = self.current_function_return_type.clone() else {
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
                        let _ = self.check_expression(expr, None);
                        self.errors.push(TypeError::ReturnValueFromVoid(span));
                    }
                    (Some(expr), expected_type) => {
                        // Check the expression and its type
                        if let Some(found_type) = self.check_expression(expr, None) {
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
    fn check_binding(
        &mut self,
        name: &str,
        annotation_type: &Option<Type>, // Pass resolved Option<Type>
        value: &Expression,
        is_mutable: bool,
        stmt_span: Span,
    ) {
        // Pass annotation type as the expected type for the value expression check
        let value_type_opt = self.check_expression(value, annotation_type.clone()); // Pass Option<Type>
        let Some(value_type) = value_type_opt else {
            return;
        }; // Error checking value

        if value_type == Type::Void {
            // Cannot infer variable type from void
            self.errors.push(TypeError::VoidAssignment(
                name.to_string(),
                value.span.clone(),
            ));
            return;
        }

        // Determine expected type & check match
        let expected_type = match annotation_type {
            Some(ann_ty) => {
                // Check if value type matches annotation
                // TODO: Handle vector literal type inference/checking against annotation
                if value_type != *ann_ty
                    && !self.check_vector_literal_assignment(
                        &value_type,
                        &ann_ty,
                        value.span.clone(),
                    )
                {
                    self.errors.push(TypeError::TypeMismatch {
                        expected: ann_ty.clone(),
                        found: value_type.clone(),
                        context: format!("variable '{}'", name),
                        span: value.span.clone(),
                    });
                }
                ann_ty
            }
            None => &value_type, // Infer from value
        };

        // Define variable
        let info = SymbolInfo {
            ty: expected_type.clone(),
            is_mutable,
            def_span: stmt_span.clone(),
        };
        if let Err(e) = self.symbol_table.define_variable(name, info) {
            self.errors.push(TypeError::VariableRedefinition {
                name: name.to_string(),
                new_loc: stmt_span,
                prev_loc: e,
            });
        }
    }

    // Recursive helper for vector assignment check (needs refinement)
    // todo: Handle empty vector literal assignment
    fn check_vector_literal_assignment(
        &self,
        value_type: &Type,
        ann_ty: &Type,
        _value_span: Span,
    ) -> bool {
        match (value_type, ann_ty) {
            // Allow assigning vec<T> lit to vec<T> var
            (Type::Vector(val_et), Type::Vector(ann_et)) => val_et == ann_et,
            // Add special case for empty literal [] assigned to vec<T>?
            // If value_type comes back as some special "UnknownVector" type?
            // This depends on how check_expression handles empty literals now.
            _ => value_type == ann_ty, // Default check
        }
    }

    /// Check an expression and return its determined Type or None if error.
    /// Errors are pushed to self.errors.
    fn check_expression(
        &mut self,
        expression: &Expression,
        expected_type: Option<Type>, // Added: Contextual expected type
    ) -> Option<Type> {
        let span = expression.span.clone();
        let maybe_type: Option<Type> = match &expression.kind {
            // ... Literals, Variable (use symbol_table.lookup_variable) ...
            ExpressionKind::FloatLiteral(_) => Some(Type::Float),
            ExpressionKind::IntLiteral(_) => Some(Type::Int),
            ExpressionKind::BoolLiteral(_) => Some(Type::Bool),
            ExpressionKind::Variable(name) => {
                self.symbol_table
                    .lookup_variable(name)
                    .map(|info| info.ty.clone()) // Return the found type
                    .or_else(|| {
                        self.errors
                            .push(TypeError::UndefinedVariable(name.clone(), span));
                        None
                    })
            }
            ExpressionKind::Assignment { target, value } => {
                // Check target first to know expected type for value
                let target_type_opt = self.check_lvalue_expression(target); // Check target validity/type
                let Some(target_type) = target_type_opt else {
                    let _ = self.check_expression(value, None); // Still check RHS for errors
                    return None;
                };

                // Check value against target type
                let value_type_opt = self.check_expression(value, Some(target_type)); // Pass target type as expected
                let Some(value_type) = value_type_opt else {
                    return None;
                }; // Error checking value
                if value_type == Type::Void {
                    self.errors.push(TypeError::VoidAssignment(
                        target.to_code(),
                        value.span.clone(),
                    ));
                    return None;
                }
                Some(value_type)
            }
            ExpressionKind::BinaryOp { op, left, right } => {
                let left_type_opt = self.check_expression(left, None);
                let right_type_opt = self.check_expression(right, None);
                match (left_type_opt, right_type_opt, op) {
                    // Arithmetic
                    (
                        Some(Type::Int),
                        Some(Type::Int),
                        BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Multiply
                        | BinaryOperator::Divide,
                    ) => Some(Type::Int),
                    (
                        Some(Type::Float),
                        Some(Type::Float),
                        BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Multiply
                        | BinaryOperator::Divide,
                    ) => Some(Type::Float),
                    // --- String Concatenation ---
                    (Some(Type::String), Some(Type::String), BinaryOperator::Add) => {
                        Some(Type::String)
                    } // '+' means concat for strings
                    // --- Errors ---
                    (Some(lt), Some(rt), _) => {
                        self.errors.push(TypeError::InvalidOperation {
                            op: format!("{}", op),
                            type_info: format!("{} and {}", lt, rt),
                            span,
                        });
                        None
                    }
                    _ => None, // Error in operand(s)
                }
            }
            ExpressionKind::ComparisonOp { op, left, right } => {
                let left_type_opt = self.check_expression(left, None);
                let right_type_opt = self.check_expression(right, None);
                match (left_type_opt, right_type_opt) {
                    (Some(lt), Some(rt)) => {
                        // Allow comparing identical types (int, float, bool, string for ==/!=)
                        if lt != rt || lt == Type::Void {
                            self.errors.push(TypeError::InvalidOperation {
                                op: format!("{}", op),
                                type_info: format!("{} and {}", lt, rt),
                                span,
                            });
                            None
                        } else if lt == Type::String
                            && !matches!(
                                op,
                                ComparisonOperator::Equal | ComparisonOperator::NotEqual
                            )
                        {
                            // String comparison only allowed for == and !=
                            self.errors.push(TypeError::InvalidOperation {
                                op: format!("{}", op),
                                type_info: format!("{} and {}", lt, rt),
                                span,
                            });
                            None
                        } else {
                            Some(Type::Bool)
                        }
                    }
                    _ => None,
                }
            }
            ExpressionKind::UnaryOp { op, operand } => {
                let operand_type_opt = self.check_expression(operand, None);
                match (op, operand_type_opt) {
                    (UnaryOperator::Negate, Some(Type::Int)) => Some(Type::Int),
                    (UnaryOperator::Negate, Some(Type::Float)) => Some(Type::Float),
                    (UnaryOperator::Not, Some(Type::Bool)) => Some(Type::Bool),
                    (UnaryOperator::Negate, Some(other)) => {
                        self.errors.push(TypeError::InvalidOperation {
                            op: format!("{}", op),
                            type_info: format!("{}", other),
                            span,
                        });
                        None
                    }
                    _ => None, // Error in operand
                }
            }
            ExpressionKind::LogicalOp { op, left, right } => {
                let left_type_opt = self.check_expression(left, None);
                let right_type_opt = self.check_expression(right, None);
                let op_str = match op {
                    LogicalOperator::And => "&&",
                    LogicalOperator::Or => "||",
                };

                match (left_type_opt, right_type_opt) {
                    (Some(Type::Bool), Some(Type::Bool)) => Some(Type::Bool), // bool && bool -> bool
                    (Some(lt), Some(rt)) => {
                        // Error if not bool or type mismatch
                        if lt != Type::Bool || rt != Type::Bool {
                            self.errors.push(TypeError::InvalidOperation {
                                op: op_str.to_string(),
                                type_info: format!("{} and {}", lt, rt), // Report actual types
                                span,
                            });
                        }
                        // If types mismatch but one *was* bool, maybe report better?
                        // For now, just report general invalid op if ! (bool, bool)
                        None
                    }
                    _ => None, // Error in operand(s)
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
                        let _ = self.check_expression(&args[0], None); /* Check arg exists/is valid */
                    }
                    // TODO: More specific type check for each print variant if needed
                    Some(Type::Void) // Print calls evaluate to Void
                }
                // --- >>> NEW: push built-in <<< ---
                else if name == "push" {
                    // Expect 2 args: push(vector, element)
                    if args.len() != 2 {
                        self.errors.push(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: 2,
                            found: args.len(),
                            call_site: span,
                        });
                        return Some(Type::Void); // Still returns Void even on error?
                    }
                    let vec_type_opt = self.check_expression(&args[0], None); // Check first arg (vector)
                    let elem_type_opt = self.check_expression(&args[1], None); // Check second arg (element)

                    // Check if first arg is vector and get element type
                    if let Some(Type::Vector(expected_elem_type)) = vec_type_opt {
                        // Check if second arg type matches element type
                        if let Some(actual_elem_type) = elem_type_opt {
                            if *expected_elem_type != actual_elem_type {
                                self.errors.push(TypeError::PushElementTypeMismatch {
                                    vec_type: Type::Vector(expected_elem_type.clone()), // Provide vec type for context
                                    elem_type: actual_elem_type,
                                    span: args[1].span.clone(),
                                });
                            }
                        } // Else: error checking element already recorded
                    } else if let Some(other_type) = vec_type_opt {
                        // First arg wasn't a vector
                        self.errors
                            .push(TypeError::PushNotVector(other_type, args[0].span.clone()));
                    } // Else: error checking vector arg already recorded

                    Some(Type::Void) // Assume push returns Void
                } else if name == "len" {
                    if args.len() != 1 {
                        self.errors.push(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: 1,
                            found: args.len(),
                            call_site: span,
                        });
                        Some(Type::Int)
                    }
                    // len returns int, return type even on error?
                    else {
                        let target_type = self.check_expression(&args[0], None);
                        match target_type {
                            // len works on Vector and String
                            Some(Type::Vector(_)) | Some(Type::String) => Some(Type::Int),
                            Some(other) => {
                                self.errors.push(TypeError::InvalidOperation {
                                    op: "len".to_string(),
                                    type_info: format!("{}", other),
                                    span,
                                });
                                None
                            }
                            None => None, // Error checking arg
                        }
                    }
                } else if name == "pop" {
                    if args.len() != 1 {
                        self.errors.push(TypeError::IncorrectArgCount {
                            func_name: name.clone(),
                            expected: 1,
                            found: args.len(),
                            call_site: span,
                        });
                        Some(Type::Void)
                    }
                    // Pop returns element type or Void on error? Let's use Void on error.
                    else {
                        let target_type = self.check_expression(&args[0], None);
                        match target_type {
                            Some(Type::Vector(elem_type)) => Some(*elem_type), // Pop returns element type
                            Some(other) => {
                                self.errors.push(TypeError::InvalidOperation {
                                    op: "pop".to_string(),
                                    type_info: format!("{}", other),
                                    span,
                                });
                                None
                            }
                            None => None, // Error checking arg
                        }
                    }
                }
                // --- User Functions ---
                else {
                    match self.symbol_table.lookup_function(name).cloned() {
                        // Clone sig
                        Some(signature) => {
                            if args.len() != signature.param_types.len() {
                                // Error: incorrect arg count
                                self.errors.push(TypeError::IncorrectArgCount {
                                    func_name: name.clone(),
                                    expected: signature.param_types.len(),
                                    found: args.len(),
                                    call_site: span,
                                });
                            }
                            // Check arg types
                            for i in 0..std::cmp::min(args.len(), signature.param_types.len()) {
                                let Some(arg_type) = self.check_expression(&args[i], None) else {
                                    continue;
                                };
                                let expected = signature.param_types[i].clone();
                                if arg_type != expected {
                                    self.errors.push(TypeError::TypeMismatch {
                                        expected: expected.clone(),
                                        found: arg_type,
                                        context: format!(
                                            "argument {} of function '{}'",
                                            i + 1,
                                            name
                                        ),
                                        span: args[i].span.clone(),
                                    });
                                }
                            }
                            Some(signature.return_type)
                        }
                        None => {
                            self.errors
                                .push(TypeError::UndefinedFunction(name.clone(), span));
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
                // Check condition bool
                match self.check_expression(condition, Some(Type::Bool)) {
                    Some(Type::Bool) | None => {} // OK or error already recorded
                    Some(other) => {
                        self.errors.push(TypeError::InvalidConditionType(
                            other,
                            condition.span.clone(),
                        ));
                    }
                }
                // Check branches against the overall expected type for the IfExpr (if any)
                let then_type_opt = self.check_expression(then_branch, expected_type.clone());
                let else_type_opt = self.check_expression(else_branch, expected_type);
                // Check types match each other and are not void
                match (then_type_opt, else_type_opt) {
                    (Some(tt), Some(et)) if tt != Type::Void && tt == et => Some(tt), // OK
                    (Some(tt), Some(et)) => {
                        // Mismatch or Void
                        self.errors.push(TypeError::IfBranchMismatch {
                            then_type: tt,
                            else_type: et,
                            span: expression.span.clone(),
                        });
                        None
                    }
                    _ => None, // Error in one or both branches
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
                    self.check_expression(final_expr, None) // Type is final expression's type (or None if error)
                } else {
                    Some(Type::Void)
                }; // No final expr -> Void
                self.symbol_table.exit_scope();
                block_type
            }
            ExpressionKind::VectorLiteral { elements } => {
                if elements.is_empty() {
                    // EMPTY literal []: Use expected type if available
                    match expected_type {
                        Some(Type::Vector(expected_elem_type)) => {
                            // OK: Empty literal matches expected vec<T>
                            Some(Type::Vector(expected_elem_type))
                        }
                        _ => {
                            // Cannot infer type, no context or context not vector
                            self.errors.push(TypeError::CannotInferVectorType(span));
                            None
                        }
                    }
                } else {
                    // NON-EMPTY literal: Infer from first, check consistency, maybe check against expected
                    let first_elem_type_opt = self.check_expression(&elements[0], None); // Check first element without expectation first?
                                                                                         // Or pass down expected *element* type if known?
                    let expected_elem_type = expected_type.as_ref().and_then(|t| match t {
                        Type::Vector(et) => Some(*et.clone()), // Deref Box<Type>
                        _ => None,
                    });

                    let Some(inferred_elem_type) =
                        self.check_expression(&elements[0], expected_elem_type.clone())
                    else {
                        return None;
                    };
                    if inferred_elem_type == Type::Void {
                        // Cannot infer vector type from void
                        self.errors
                            .push(TypeError::CannotInferVectorType(elements[0].span.clone()));
                        return None;
                    }

                    // If context provided expected element type, ensure inferred type matches
                    if let Some(expected_et) = expected_elem_type.clone() {
                        if inferred_elem_type != expected_et {
                            self.errors.push(TypeError::VectorElementTypeMismatch {
                                expected: expected_et,
                                found: inferred_elem_type,
                                span: elements[0].span.clone(),
                            });
                            return None;
                        }
                    }

                    // Check remaining elements match inferred/expected type
                    for element_expr in elements.iter().skip(1) {
                        let current_elem_type_opt =
                            self.check_expression(element_expr, Some(inferred_elem_type.clone())); // Expect consistent type
                        if current_elem_type_opt != Some(inferred_elem_type.clone()) {
                            // Error already pushed by check_expression if mismatch against expected,
                            // or if element check itself failed (returned None)
                            return None;
                        }
                    }
                    Some(Type::Vector(Box::new(inferred_elem_type))) // Return inferred Vector type
                }
            }
            // --- Index Access ---
            ExpressionKind::IndexAccess { target, index } => {
                let target_type_opt = self.check_expression(target, None);
                let index_type_opt = self.check_expression(index, None);

                // Check index is Int
                if let Some(it) = index_type_opt {
                    if it != Type::Int {
                        self.errors
                            .push(TypeError::IndexNotInteger(it, index.span.clone()));
                    }
                } // else error already recorded

                // Check target type and return element type
                match target_type_opt {
                    Some(Type::Vector(elem_type)) => {
                        // Indexing vec<T> yields T
                        Some(*elem_type) // <<< This already handles nested vectors correctly!
                                         // If target_type is vec<vec<int>>, elem_type is vec<int>,
                                         // so this access returns vec<int>. A subsequent index access
                                         // will then see the target as vec<int> and return int.
                    }
                    Some(Type::String) => {
                        // Allow string indexing? Return char type? Need Type::Char.
                        // self.errors
                        //     .push(TypeError::NotYetImplemented("String indexing".into(), span));
                        todo!("String indexing");
                        None // String indexing not fully supported yet
                    }
                    Some(other_type) => {
                        self.errors
                            .push(TypeError::NotAVector(other_type, target.span.clone()));
                        None
                    }
                    None => None, // Error checking target
                }
            } // End IndexAccess
            _ => unimplemented!(
                "Type checking not implemented for this expression node yet: {:?}",
                expression
            ),
        };
        // --- Annotation ---
        // If we successfully determined a type, store it in the AST node
        if let Some(ty) = maybe_type.clone() {
            expression.set_type(ty); // Use helper to set type in RefCell
        }

        maybe_type // Return the determined type (or None if error occurred)
    } // End check_expression

    // New helper to check expressions used as L-values (targets of assignment)
    // Returns the type of the target if valid, pushes errors otherwise.
    fn check_lvalue_expression(&mut self, target: &Expression) -> Option<Type> {
        match &target.kind {
            ExpressionKind::Variable(name) => {
                let info_opt = self.symbol_table.lookup_variable(name);
                match info_opt {
                    Some(info) => {
                        if !info.is_mutable {
                            self.errors.push(TypeError::AssignmentToImmutable(
                                name.clone(),
                                target.span.clone(),
                            ));
                            None // Invalid l-value if immutable
                        } else {
                            Some(info.ty.clone())
                        } // Valid l-value, return its type
                    }
                    None => {
                        self.errors.push(TypeError::UndefinedVariable(
                            name.clone(),
                            target.span.clone(),
                        ));
                        None
                    }
                }
            }
            // Case 2: Index Access - e.g., vec[idx] or matrix[i][j]
            ExpressionKind::IndexAccess {
                target: inner_target,
                index,
            } => {
                // Index must be integer
                match self.check_expression(index, Some(Type::Int)) {
                    Some(Type::Int) => {} // OK
                    Some(other) => {
                        self.errors
                            .push(TypeError::IndexNotInteger(other, index.span.clone()));
                        return None;
                    }
                    None => return None, // Error checking index
                }

                // --- Check the INNER target ---
                // The inner target (e.g., 'vec' in vec[idx], or 'matrix[i]' in matrix[i][j])
                // must evaluate to a vector, AND must itself be mutable *at this level*
                // (e.g., if matrix is `let`, matrix[i][j]=... fails).
                // However, checking the mutability of the *result* of inner_target isn't quite right.
                // We need to check if the base variable (e.g., 'matrix') is mutable.

                // Let's recursively check the *inner target* as an L-Value first? No, that's for target[i][j] = ...
                // For target[i][j] = value, we need the *type* of target[i] and mutability of target.

                // Check type of inner_target (e.g. matrix[i])
                let Some(inner_target_type) = self.check_expression(inner_target, None) else {
                    return None;
                };

                match inner_target_type {
                    Type::Vector(elem_type) => {
                        // The inner target (e.g. matrix[i]) resolved to a vector.
                        // Now, check if the base of this access (e.g., "matrix") is mutable.
                        // This requires traversing down the L-value chain.
                        if !self.is_lvalue_base_mutable(inner_target) {
                            self.errors.push(TypeError::AssignmentToImmutable(
                                "vector element via immutable base".to_string(), // Improve msg
                                inner_target.span.clone(),
                            ));
                            return None;
                        }
                        // The L-value `target[i][j]` has the type of the element.
                        Some(*elem_type)
                    }
                    // Add String indexing assignment check later?
                    _ => {
                        self.errors.push(TypeError::AssignIndexTargetNotVector(
                            inner_target_type,
                            inner_target.span.clone(),
                        ));
                        None
                    }
                }
            } // End IndexAccess case
            // Invalid L-value kind
            _ => {
                self.errors.push(TypeError::InvalidAssignmentTarget(
                    target.to_code(),
                    target.span.clone(),
                ));
                None
            }
        }
    }

    // Helper to recursively check if the base variable of an L-Value expression is mutable
    fn is_lvalue_base_mutable(&self, expr: &Expression) -> bool {
        match &expr.kind {
            ExpressionKind::Variable(name) => {
                // Base case: check the variable itself
                self.symbol_table.lookup_variable(name).map_or(false, |info| info.is_mutable)
                // If lookup fails, UndefinedVariable error handled elsewhere, return false here
            }
            ExpressionKind::IndexAccess { target, .. } => {
                // Recursive case: check the target of the index access
                self.is_lvalue_base_mutable(target)
            }
            // Add cases for field access later (e.g., struct.field) -> return self.is_lvalue_base_mutable(struct_expr)
            _ => false, // Other expression kinds are not mutable L-values bases
        }
    }

    // Helper to resolve TypeNode - needs self now to push errors
    fn resolve_type_node(&mut self, node: &TypeNode) -> Option<Type> {
        match &node.kind {
            TypeNodeKind::Simple(name) => match name.as_str() {
                "int" => Some(Type::Int),
                "float" => Some(Type::Float),
                "bool" => Some(Type::Bool),
                "string" => Some(Type::String), // Handle string type name
                "void" => Some(Type::Void),
                // Add more types as needed
                _ => {
                    self.errors.push(TypeError::UnknownTypeName(
                        name.clone(),
                        node.span.start.clone(),
                    ));
                    None
                }
            },
            TypeNodeKind::Vector(element_node) => {
                // Recursively resolve element type
                self.resolve_type_node(element_node)
                    .map(|t| Type::Vector(Box::new(t)))
            }
        }
    }

    // Helper to check blocks used inside statements like IfStmt, WhileStmt, ForStmt
    // These blocks don't produce a value for the statement itself.
    fn check_program_block(&mut self, program: &Program) {
        // Simply check statements, collecting errors in self.errors
        for stmt in &program.statements {
            self.check_statement(stmt);
        }
    }
} // End impl TypeChecker

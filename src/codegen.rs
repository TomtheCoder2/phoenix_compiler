// src/codegen.rs

use crate::ast::{BinaryOperator, Expression, Program, Statement};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{AnyValue, BasicValueEnum, FloatValue, FunctionValue, PointerValue};
// Added BasicValueEnum
use std::collections::HashMap;
use std::fmt;

// --- CodeGenError --- (UndefinedVariable is still relevant)
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenError {
    InvalidAstNode(String),
    LlvmError(String),
    // UnknownOperator(BinaryOperator), // Covered by expression compilation
    UndefinedVariable(String),
    // Maybe add specific errors for codegen issues
}
impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenError::InvalidAstNode(msg) => write!(f, "Invalid AST Node: {}", msg),
            CodeGenError::LlvmError(msg) => write!(f, "LLVM Error: {}", msg),
            CodeGenError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
        }
    }
}
type CompileResult<'ctx, T> = Result<T, CodeGenError>; // Generic result

pub struct Compiler<'a, 'ctx> {
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    variables: HashMap<String, PointerValue<'ctx>>, // Symbol table persists across statements
    current_function: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    pub fn new(
        context: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
    ) -> Self {
        Compiler {
            context,
            builder,
            module,
            variables: HashMap::new(),
            current_function: None,
        }
    }

    // Helper to create an alloca instruction in the function's entry block
    // This ensures all allocas happen at the start, which is good practice in LLVM.
    fn create_entry_block_alloca(&self, name: &str) -> PointerValue<'ctx> {
        // Create a temporary builder positioned at the beginning of the function's entry block
        let temp_builder = self.context.create_builder();
        let entry_block = self
            .current_function
            .unwrap()
            .get_first_basic_block()
            .unwrap();

        match entry_block.get_first_instruction() {
            Some(first_instr) => temp_builder.position_before(&first_instr),
            None => temp_builder.position_at_end(entry_block), // If block is empty
        };

        // Allocate memory for an f64 on the stack
        let f64_type = self.context.f64_type();
        // todo: Handle potential errors in build_alloca
        temp_builder.build_alloca(f64_type, name).unwrap()
    }

    /// Compiles a single Expression node (used by statements)
    /// Returns a FloatValue representing the result of the expression.
    fn compile_expression(&mut self, expr: &Expression) -> CompileResult<'ctx, FloatValue<'ctx>> {
        match expr {
            Expression::NumberLiteral(value) => {
                let f64_type = self.context.f64_type();
                Ok(f64_type.const_float(*value))
            }
            Expression::Variable(name) => {
                match self.variables.get(name) {
                    Some(var_ptr) => {
                        let f64_type = self.context.f64_type();
                        // build_load now returns BasicValueEnum wrapped in Result
                        let loaded_val_result = self.builder.build_load(f64_type, *var_ptr, name);
                        match loaded_val_result {
                            Ok(BasicValueEnum::FloatValue(fv)) => Ok(fv),
                            Ok(_) => Err(CodeGenError::LlvmError(format!(
                                "Expected FloatValue from load for var '{}'",
                                name
                            ))),
                            Err(e) => Err(CodeGenError::LlvmError(format!(
                                "LLVM build_load error for var '{}': {}",
                                name, e
                            ))),
                        }
                    }
                    None => Err(CodeGenError::UndefinedVariable(name.clone())),
                }
            }
            Expression::BinaryOp { op, left, right } => {
                let lhs = self.compile_expression(left)?;
                let rhs = self.compile_expression(right)?;
                match match op {
                    BinaryOperator::Add => self.builder.build_float_add(lhs, rhs, "addtmp"),
                    BinaryOperator::Subtract => self.builder.build_float_sub(lhs, rhs, "subtmp"),
                    BinaryOperator::Multiply => self.builder.build_float_mul(lhs, rhs, "multmp"),
                    BinaryOperator::Divide => self.builder.build_float_div(lhs, rhs, "divtmp"),
                } {
                    Ok(value) => Ok(value),
                    Err(e) => {
                        // Handle potential LLVM errors
                        Err(CodeGenError::LlvmError(format!(
                            "LLVM error during binary operation: {}",
                            e
                        )))
                    }
                }
            }
        }
    }

    /// Compiles a single Statement node.
    /// Returns Ok(Some(FloatValue)) if it's an ExpressionStmt, Ok(None) for LetBinding, Err on failure.
    fn compile_statement(
        &mut self,
        stmt: &Statement,
    ) -> CompileResult<'ctx, Option<FloatValue<'ctx>>> {
        match stmt {
            Statement::LetBinding { name, value } => {
                // Compile the value expression first
                let compiled_value = self.compile_expression(value)?;

                // Check if variable already exists (for shadowing or reassignment later)
                let alloca = if let Some(existing_alloca) = self.variables.get(name) {
                    // Variable already exists (shadowing in current scope, or reassignment if mutable)
                    // For now, we just reuse the allocation. If we add proper scopes/mutability,
                    // this logic needs refinement.
                    *existing_alloca
                } else {
                    // Allocate memory for the new variable
                    let new_alloca = self.create_entry_block_alloca(name);
                    self.variables.insert(name.clone(), new_alloca); // Add to symbol table
                    new_alloca
                };

                // Store the compiled value into the allocated memory
                self.builder.build_store(alloca, compiled_value);

                // Let statements don't produce a value themselves in this model
                Ok(None)
            }

            Statement::ExpressionStmt(expr) => {
                // Compile the expression
                let expr_value = self.compile_expression(expr)?;
                // This statement *does* produce a value
                Ok(Some(expr_value))
            }
        }
    }

    /// Compiles the entire Program (list of statements) into an LLVM function.
    /// The function will return the value of the *last* ExpressionStmt encountered.
    pub fn compile_program(
        &mut self,
        program: &Program,
    ) -> Result<FunctionValue<'ctx>, CodeGenError> {
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[], false);
        let function = self.module.add_function("main", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);
        self.current_function = Some(function);
        self.variables.clear(); // Clear variables for this compilation unit

        let mut last_expr_value: Option<FloatValue<'ctx>> = None;

        // Compile statements one by one
        for stmt in &program.statements {
            match self.compile_statement(stmt) {
                Ok(Some(value)) => last_expr_value = Some(value), // Capture value from ExpressionStmt
                Ok(None) => {} // LetBinding doesn't change the last value
                Err(e) => {
                    // Error during statement compilation: cleanup and return error
                    unsafe {
                        function.delete();
                    }
                    self.current_function = None;
                    return Err(e);
                }
            }
        }

        // Build the return instruction
        if let Some(return_value) = last_expr_value {
            self.builder.build_return(Some(&return_value));
        } else {
            // No expression statement found, or program was empty. Return 0.0? Or error?
            // Let's return 0.0 for now. Could also make main return void.
            self.builder.build_return(Some(&f64_type.const_float(0.0)));
        }

        self.current_function = None;

        // Verify
        if function.verify(true) {
            Ok(function)
        } else {
            eprintln!(
                "Invalid function generated (verify failed):\n{}",
                function.print_to_string().to_string()
            );
            // unsafe { function.delete(); } // Optionally delete malformed function
            Err(CodeGenError::LlvmError(
                "Function verification failed".to_string(),
            ))
        }
    }
}

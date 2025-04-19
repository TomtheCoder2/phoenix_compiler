// src/codegen.rs

use crate::ast::{BinaryOperator, Expression};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
// Added PointerType
use inkwell::values::{AnyValue, FloatValue, FunctionValue, PointerValue};
// Added PointerValue
use std::collections::HashMap;
use std::fmt;
// Use HashMap for symbol table

// --- CompileResult and CodeGenError remain similar ---
// Add error variant for undefined variable
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenError {
    InvalidAstNode(String),
    LlvmError(String),
    UnknownOperator(BinaryOperator),
    UndefinedVariable(String), // Added
}

impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenError::InvalidAstNode(msg) => write!(f, "Invalid AST Node: {}", msg),
            CodeGenError::LlvmError(msg) => write!(f, "LLVM Error: {}", msg),
            CodeGenError::UnknownOperator(op) => write!(f, "Unknown binary operator: {:?}", op),
            CodeGenError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
        }
    }
}
type CompileResult<'ctx> = Result<FloatValue<'ctx>, CodeGenError>;

pub struct Compiler<'a, 'ctx> {
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    // fpm: &'a PassManager<FunctionValue<'ctx>>, // Optimization passes manager

    // Symbol Table: Map variable names (String) to their memory location (PointerValue)
    variables: HashMap<String, PointerValue<'ctx>>,
    // Keep track of the function currently being built is useful for allocas
    current_function: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    pub fn new(
        context: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
        // fpm: &'a PassManager<FunctionValue<'ctx>>,
    ) -> Self {
        Compiler {
            context,
            builder,
            module,
            // fpm,
            variables: HashMap::new(), // Initialize empty symbol table
            current_function: None,    // No function initially
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

    /// Compiles an AST Expression node into an LLVM FloatValue
    fn compile_expr(&mut self, expr: &Expression) -> CompileResult<'ctx> {
        match expr {
            Expression::NumberLiteral(value) => {
                let f64_type = self.context.f64_type();
                Ok(f64_type.const_float(*value))
            }

            Expression::Variable(name) => {
                match self.variables.get(name) {
                    Some(var_ptr) => {
                        // Load the value from the variable's memory location
                        let f64_type = self.context.f64_type();
                        let loaded_val = self.builder.build_load(f64_type, *var_ptr, name);
                        if let Err(e) = loaded_val {
                            return Err(CodeGenError::LlvmError(format!(
                                "Failed to load variable '{}': {}",
                                name, e
                            )));
                        }
                        // has to be non-null
                        let loaded_val = loaded_val.unwrap();
                        // build_load returns Result<BasicValueEnum, String>, we need FloatValue
                        // Assuming it loads correctly and is a float value
                        Ok(loaded_val.into_float_value())
                    }
                    None => Err(CodeGenError::UndefinedVariable(name.clone())),
                }
            }

            Expression::BinaryOp { op, left, right } => {
                let lhs = self.compile_expr(left)?;
                let rhs = self.compile_expr(right)?;

                match match op {
                    BinaryOperator::Add => self.builder.build_float_add(lhs, rhs, "addtmp"),
                    BinaryOperator::Subtract => self.builder.build_float_sub(lhs, rhs, "subtmp"),
                    BinaryOperator::Multiply => self.builder.build_float_mul(lhs, rhs, "multmp"),
                    BinaryOperator::Divide => self.builder.build_float_div(lhs, rhs, "divtmp"),
                    // Note: We might add comparison operators later which return i1 (bool)
                    // _ => Err(CodeGenError::UnknownOperator(*op)), // Handle if necessary
                } {
                    Ok(fv) => Ok(fv),
                    Err(e) => CompileResult::Err(CodeGenError::LlvmError(e.to_string())),
                }
            }

            Expression::Let { name, value, body } => {
                // 1. Compile the value expression
                let compiled_value = self.compile_expr(value)?;

                // 2. Allocate memory (alloca) for the variable in the function entry block
                let alloca = self.create_entry_block_alloca(name);

                // 3. Store the compiled value into the allocated memory
                self.builder.build_store(alloca, compiled_value);

                // --- Scoping ---
                // 4. Temporarily bind the variable name to its alloca in the symbol table.
                //    Handle potential shadowing: store old value if exists.
                let old_binding = self.variables.insert(name.clone(), alloca);

                // 5. Compile the body expression. The variable 'name' is now visible.
                let body_result = self.compile_expr(body); // Result<FloatValue, Error>

                // 6. Restore the symbol table: Remove the binding or restore the old one.
                if let Some(old_ptr) = old_binding {
                    self.variables.insert(name.clone(), old_ptr); // Restore shadowed variable
                } else {
                    self.variables.remove(name); // Remove if it wasn't shadowing
                }
                // --- End Scoping ---

                // 7. Return the result of compiling the body
                body_result
            }
        }
    }

    // Compile the top-level expression into a main function
    pub fn compile(&mut self, ast: &Expression) -> Result<FunctionValue<'ctx>, CodeGenError> {
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[], false);
        let function = self.module.add_function("main", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);
        self.current_function = Some(function); // Set the current function

        // Clear variables map for this function compilation
        self.variables.clear();

        // Compile the main expression body
        let body = match self.compile_expr(ast) {
            Ok(val) => val,
            Err(e) => {
                // If compilation fails, it's good practice to delete the potentially malformed function
                unsafe {
                    function.delete();
                }
                self.current_function = None;
                return Err(e);
            }
        };

        // Build the return instruction
        self.builder.build_return(Some(&body));

        // Reset current function
        self.current_function = None;

        // Verify the function
        if function.verify(true) {
            Ok(function)
        } else {
            eprintln!(
                "Invalid function generated (potentially due to error during build):\n{}",
                function.print_to_string().to_string()
            );
            // Consider deleting the function if verification fails after successful compilation?
            // unsafe { function.delete(); }
            Err(CodeGenError::LlvmError(
                "Function verification failed".to_string(),
            ))
        }
    }
}

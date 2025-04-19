// src/codegen.rs

use crate::ast::{BinaryOperator, Expression};
// For comparisons later
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{AnyValue, FloatValue, FunctionValue};
// For symbol table (variables) later

// A helper type alias for our recursive compile function result
type CompileResult<'ctx> = Result<FloatValue<'ctx>, CodeGenError>;

// Define potential code generation errors
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenError {
    InvalidAstNode(String), // Indicates an AST node we can't handle yet
    LlvmError(String),      // For errors originating from LLVM operations
    UnknownOperator(BinaryOperator), // Operator not handled in codegen
                            // Add more specific errors as needed
}

// Implement Display for nice error messages
use std::fmt;
impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenError::InvalidAstNode(msg) => write!(f, "Invalid AST Node: {}", msg),
            CodeGenError::LlvmError(msg) => write!(f, "LLVM Error: {}", msg),
            CodeGenError::UnknownOperator(op) => write!(f, "Unknown binary operator: {:?}", op),
        }
    }
}

// The main struct for handling code generation
pub struct Compiler<'a, 'ctx> {
    context: &'ctx Context, // The LLVM context (owns core LLVM data structures)
    builder: &'a Builder<'ctx>, // Helper to construct LLVM instructions
    module: &'a Module<'ctx>, // The LLVM module (contains functions, globals, etc.)
                            // fpm: &'a PassManager<FunctionValue<'ctx>>, // Function Pass Manager for optimizations (add later)

                            // Symbol table for variables (we'll need this soon)
                            // variables: HashMap<String, PointerValue<'ctx>>,
                            // current_function: Option<FunctionValue<'ctx>>, // Keep track of the function being built
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    // Note: Constructor is slightly simplified for now.
    // A more complete version would likely create the context, builder, module internally.
    // For now, we assume they are passed in.
    pub fn new(
        context: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
        // fpm: &'a PassManager<FunctionValue<'ctx>>, // Pass FPM later
    ) -> Self {
        Compiler {
            context,
            builder,
            module,
            // fpm,
            // variables: HashMap::new(),
            // current_function: None,
        }
    }

    /// Compiles an AST Expression node into an LLVM FloatValue
    fn compile_expr(&mut self, expr: &Expression) -> CompileResult<'ctx> {
        match expr {
            Expression::NumberLiteral(value) => {
                // Get the f64 type from the LLVM context
                let f64_type = self.context.f64_type();
                // Create an LLVM constant float value
                let llvm_value = f64_type.const_float(*value);
                Ok(llvm_value)
            }

            Expression::BinaryOp { op, left, right } => {
                // Recursively compile the left and right sub-expressions
                let lhs = self.compile_expr(left)?;
                let rhs = self.compile_expr(right)?;

                // Use the builder to create the appropriate float instruction
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
            } // Handle other expression types (variables, calls) later
              // _ => Err(CodeGenError::InvalidAstNode(format!("Unsupported expression type: {:?}", expr))),
        }
    }

    /// Compiles the entire AST, creating a main function to wrap the expression.
    pub fn compile(&mut self, ast: &Expression) -> Result<FunctionValue<'ctx>, CodeGenError> {
        // Define the function type: f64() (takes no arguments, returns f64)
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[], false); // No params, not variadic

        // Add the function to the module
        let function = self.module.add_function("main", fn_type, None); // Use "main" for simplicity

        // Create a basic block within the function to hold instructions
        let basic_block = self.context.append_basic_block(function, "entry");

        // Position the builder at the end of the new basic block
        self.builder.position_at_end(basic_block);

        // --- Compile the expression ---
        let body = self.compile_expr(ast)?;

        // --- Build the return instruction ---
        self.builder.build_return(Some(&body));

        // --- (Optional but recommended) Verify the function ---
        // This checks for structural errors in the generated IR
        if function.verify(true) {
            Ok(function)
        } else {
            // Often good to print the IR on verification failure
            // unsafe { function.delete(); } // Clean up the malformed function
            eprintln!(
                "Invalid function generated:\n{}",
                function.print_to_string().to_string()
            );
            Err(CodeGenError::LlvmError(
                "Function verification failed".to_string(),
            ))
        }

        // --- (Optional) Run optimization passes ---
        // self.fpm.run_on(&function); // Add this later
    }
}

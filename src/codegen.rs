// src/codegen.rs

use crate::ast::{BinaryOperator, Expression, Program, Statement};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{AnyValue, BasicValueEnum, FloatValue, FunctionValue, PointerValue};
// Added BasicValueEnum
use std::collections::HashMap;
use std::fmt;
use inkwell::types::BasicMetadataTypeEnum;

// --- CodeGenError --- (UndefinedVariable is still relevant)
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenError {
    InvalidAstNode(String),
    LlvmError(String),
    // UnknownOperator(BinaryOperator), // Covered by expression compilation
    UndefinedVariable(String),
    UndefinedFunction(String),    // Added
    FunctionRedefinition(String), // Added
    IncorrectArgumentCount {
        // Added
        func_name: String,
        expected: usize,
        found: usize,
    },
}
impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenError::InvalidAstNode(msg) => write!(f, "Invalid AST Node: {}", msg),
            CodeGenError::LlvmError(msg) => write!(f, "LLVM Error: {}", msg),
            CodeGenError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            CodeGenError::UndefinedFunction(name) => {
                write!(f, "Codegen Error: Undefined function '{}'", name)
            }
            CodeGenError::FunctionRedefinition(name) => {
                write!(f, "Codegen Error: Function '{}' redefined", name)
            }
            CodeGenError::IncorrectArgumentCount {
                func_name,
                expected,
                found,
            } => write!(
                f,
                "Codegen Error: Call to function '{}' expected {} arguments, found {}",
                func_name, expected, found
            ),
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
    functions: HashMap<String, FunctionValue<'ctx>>, // Stores defined functions
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
            functions: HashMap::new(), // Initialize function map
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
            // --- Handle Function Calls ---
            Expression::FunctionCall { name, args } => {
                // 1. Look up the function in our function map
                let func_val_opt = self.functions.get(name); // Immutable borrow starts

                // --- FIX START ---
                // Clone the FunctionValue immediately if found. FunctionValue is Copy.
                // This ends the immutable borrow of self.functions needed for the lookup.
                let func_to_call = match func_val_opt {
                    Some(f) => *f, // Dereference to get FunctionValue and copy it
                    None => return Err(CodeGenError::UndefinedFunction(name.clone())),
                };

                // 2. Check argument count USING THE CLONED VALUE
                let expected_count = func_to_call.count_params() as usize; // Use cloned func_to_call
                if args.len() != expected_count {
                    return Err(CodeGenError::IncorrectArgumentCount {
                        func_name: name.clone(),
                        expected: expected_count,
                        found: args.len(),
                    });
                }

                // 3. Compile each argument expression
                // NOW, this mutable borrow of `self` is fine, because the immutable
                // borrow needed for the function lookup is finished.
                let mut compiled_args = Vec::with_capacity(args.len());
                for arg_expr in args {
                    let arg_val = self.compile_expression(arg_expr)?; // Mutable borrow here is OK
                    compiled_args.push(arg_val.into());
                }

                // 4. Build the call instruction
                let call_site_val = match self
                    .builder
                    .build_call(func_to_call, &compiled_args, "calltmp") {
                    Ok(call_site_val) => call_site_val,
                    Err(e) => {
                        // Handle potential LLVM errors
                        return Err(CodeGenError::LlvmError(format!(
                            "LLVM error during function call '{}': {}",
                            name, e
                        )));
                    }
                };

                // The result of build_call is Result<CallSiteValue, ErrorMsg>
                // CallSiteValue represents the return value (or void).
                // We need to try converting it back to a FloatValue.
                match call_site_val.try_as_basic_value().left() {
                    Some(BasicValueEnum::FloatValue(fv)) => Ok(fv),
                    _ => Err(CodeGenError::LlvmError(format!(
                        "Call to '{}' did not return a FloatValue",
                        name
                    ))),
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

            // --- Handle Function Definitions ---
            Statement::FunctionDef { name, params, body } => {
                // Check for redefinition
                if self.functions.contains_key(name) {
                    return Err(CodeGenError::FunctionRedefinition(name.clone()));
                }

                // --- Define Function Type ---
                let f64_type = self.context.f64_type();
                // Create a vector of f64 types for parameters
                let param_types: Vec<BasicMetadataTypeEnum> =
                    std::iter::repeat(f64_type.into())
                        .take(params.len())
                        .collect();
                let fn_type = f64_type.fn_type(&param_types, false); // f64(f64, f64, ...)

                // --- Create LLVM Function ---
                let function = self.module.add_function(name, fn_type, None);

                // --- Store Function Value ---
                // Store *before* compiling body in case of recursion (though not fully supported yet)
                self.functions.insert(name.clone(), function);

                // --- Setup Function Body Context ---
                let entry_block = self.context.append_basic_block(function, "entry");
                let original_builder_pos = self.builder.get_insert_block(); // Save current position
                let original_func = self.current_function;
                let original_vars = self.variables.clone(); // Save current variables

                self.builder.position_at_end(entry_block); // Move builder to new function
                self.current_function = Some(function);
                self.variables.clear(); // Use fresh variables map for function scope

                // --- Allocate and Store Parameters ---
                for (i, param_name) in params.iter().enumerate() {
                    let llvm_param = function.get_nth_param(i as u32).unwrap();
                    llvm_param.set_name(param_name); // Set name in IR

                    // Allocate space for the parameter on the stack
                    let param_alloca = self.create_entry_block_alloca(param_name);
                    // Store the incoming parameter value into its stack slot
                    self.builder.build_store(param_alloca, llvm_param);
                    // Add the parameter to the function's local variables map
                    self.variables.insert(param_name.clone(), param_alloca);
                }

                // --- Compile Function Body Statements ---
                let mut last_body_val: Option<FloatValue<'ctx>> = None;
                let mut body_compile_err: Option<CodeGenError> = None;

                for body_stmt in &body.statements {
                    match self.compile_statement(body_stmt) {
                        Ok(Some(val)) => last_body_val = Some(val),
                        Ok(None) => {}
                        Err(e) => {
                            body_compile_err = Some(e);
                            break; // Stop compiling body on error
                        }
                    }
                }

                // --- Build Return ---
                // Only build return if body compilation didn't fail
                if body_compile_err.is_none() {
                    if let Some(ret_val) = last_body_val {
                        self.builder.build_return(Some(&ret_val));
                    } else {
                        // No expression statement, return default 0.0
                        self.builder.build_return(Some(&f64_type.const_float(0.0)));
                    }
                }

                // --- Restore Outer Context ---
                self.variables = original_vars; // Restore caller's variables
                self.current_function = original_func;
                if let Some(original_block) = original_builder_pos {
                    self.builder.position_at_end(original_block); // Restore builder position
                } else {
                    // Handle case where builder wasn't positioned before (e.g., first function)
                    // Maybe self.builder.clear_insertion_position(); ? Or leave it?
                }


                // --- Handle Body Compilation Error ---
                if let Some(err) = body_compile_err {
                    // Function definition itself failed, remove from map and module?
                    self.functions.remove(name);
                    unsafe { function.delete(); } // Delete potentially malformed function
                    return Err(err);
                }

                // --- Verification ---
                if function.verify(true) {
                    // Function definition doesn't produce a "value" for the outer scope
                    Ok(None)
                } else {
                    eprintln!("Invalid function generated '{}':\n{}", name, function.print_to_string().to_string());
                    self.functions.remove(name); // Remove bad function from our map
                    unsafe { function.delete(); } // Delete from module
                    Err(CodeGenError::LlvmError(format!("Function '{}' verification failed", name)))
                }
            } // End FunctionDef
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

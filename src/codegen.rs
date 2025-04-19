// src/codegen.rs

use crate::ast::{BinaryOperator, Expression, Program, Statement};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::passes::{PassManager, PassManagerBuilder};
// Added BasicValueEnum
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::values::BasicValue;
use inkwell::values::{
    AnyValue, BasicValueEnum, FloatValue, FunctionValue, GlobalValue, PointerValue,
};
use inkwell::{AddressSpace, OptimizationLevel};
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::path::Path;

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
    PrintArgError(String),
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
            CodeGenError::PrintArgError(msg) => write!(f, "Built-in 'print' Error: {}", msg),
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
    // Add the Function Pass Manager
    fpm: PassManager<FunctionValue<'ctx>>,
    // printf_func: Option<FunctionValue<'ctx>>, // Cache printf function value
    print_wrapper_func: Option<FunctionValue<'ctx>>, // Cache for our wrapper
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    pub fn new(
        context: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
    ) -> Self {
        // --- Initialize the Pass Manager ---
        let pass_manager_builder = PassManagerBuilder::create();
        // Configure standard optimization passes based on a level.
        // Level 2 (`Default`) is a good balance. Level 3 (`Aggressive`) is more intense.
        pass_manager_builder.set_optimization_level(OptimizationLevel::Default);

        // Create the Function Pass Manager
        let fpm = PassManager::create(module); // PassManager::create requires module for Module passes

        // Add desired passes - PassManagerBuilder helps populate common ones
        // You can also add passes manually using fpm.add_..._pass() methods.
        // See inkwell docs / LLVM pass list for available passes.
        // Standard sequence for O2/O3 often includes:
        // - mem2reg: Promotes memory variables (allocas) to SSA registers - HUGE gains
        // - instcombine: Combines redundant instructions
        // - gvn: Global Value Numbering (eliminates redundant calculations)
        // - simplifycfg: Simplifies control flow graph
        // - ... and many more

        // Populate the FPM using the builder's recommendations for the chosen opt level
        pass_manager_builder.populate_function_pass_manager(&fpm);

        // It's crucial to initialize the FPM *after* adding passes.
        fpm.initialize(); // Call initialize AFTER passes are added

        Compiler {
            context,
            builder,
            module,
            variables: HashMap::new(),
            functions: HashMap::new(),
            current_function: None,
            fpm, // Store the initialized FPM
            print_wrapper_func: None,
        }
    }

    // Helper to declare or get the printf function
    // Helper to declare the wrapper function
    fn get_or_declare_print_wrapper(&mut self) -> FunctionValue<'ctx> {
        if let Some(func) = self.print_wrapper_func {
            return func;
        }
        // Wrapper signature: void print_f64_wrapper(double)
        let f64_type = self.context.f64_type();
        let void_type = self.context.void_type(); // Wrapper returns void
        let wrapper_type = void_type.fn_type(&[f64_type.into()], false); // Not variadic

        let wrapper_func = self.module.add_function(
            "print_f64_wrapper", // Name must match #[no_mangle] symbol
            wrapper_type,
            Some(Linkage::External),
        );
        self.print_wrapper_func = Some(wrapper_func);
        wrapper_func
    }

    // Helper to create a global string constant (for format strings)
    fn create_global_string(&self, name: &str, value: &str, linkage: Linkage) -> GlobalValue<'ctx> {
        let c_string = CString::new(value).expect("CString::new failed");
        // Use `to_bytes()` (length *without* null) and let LLVM add the null terminator (`false`)
        // This ensures the type has the correct size (e.g., [4 x i8] for "%f\n")
        let string_val = self.context.const_string(c_string.to_bytes(), false);

        let global = self.module.add_global(
            string_val.get_type(), // Type from const_string should be correct now
            Some(AddressSpace::default()),
            name,
        );
        global.set_linkage(linkage);
        global.set_initializer(&string_val);
        global.set_constant(true);
        global
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
                // --- >> SPECIAL CASE: Built-in print function << ---
                if name == "print" {
                    // 1. Check argument count
                    if args.len() != 1 { /* ... error ... */ }

                    // 2. Compile the argument
                    let arg_value = self.compile_expression(&args[0])?;

                    // 3. Get the wrapper function
                    let wrapper_func = self.get_or_declare_print_wrapper();

                    // 4. Build the call to the WRAPPER (non-variadic)
                    self.builder.build_call(
                        wrapper_func,
                        &[arg_value.into()], // Pass only the double argument
                        "",                  // No return value assigned from call
                    );

                    // 5. Return 0.0 for the print expression
                    Ok(self.context.f64_type().const_float(0.0))

                // --- >> REGULAR FUNCTION CALL (User-defined) << ---
                } else {
                    // Look up the function in *user-defined* functions
                    let func_to_call = match self.functions.get(name) {
                        Some(f) => *f, // Copy the FunctionValue
                        None => return Err(CodeGenError::UndefinedFunction(name.clone())),
                    };

                    // Check argument count
                    let expected_count = func_to_call.count_params() as usize;
                    if args.len() != expected_count {
                        return Err(CodeGenError::IncorrectArgumentCount {
                            func_name: name.clone(),
                            expected: expected_count,
                            found: args.len(),
                        });
                    }

                    // Compile arguments
                    let mut compiled_args = Vec::with_capacity(args.len());
                    for arg_expr in args {
                        let arg_val = self.compile_expression(arg_expr)?;
                        compiled_args.push(arg_val.into());
                    }

                    // Build the call
                    let call_site_val =
                        match self
                            .builder
                            .build_call(func_to_call, &compiled_args, "calltmp")
                        {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(format!(
                                    "LLVM error during function call '{}': {}",
                                    name, e
                                )));
                            }
                        };

                    // Handle return value
                    match call_site_val.try_as_basic_value().left() {
                        Some(BasicValueEnum::FloatValue(fv)) => Ok(fv),
                        _ => Err(CodeGenError::LlvmError(format!(
                            "Call to user function '{}' did not return a FloatValue",
                            name
                        ))),
                    }
                } // End if name == "print" / else
            } // End FunctionCall
        }
    }

    /// Compiles a single Statement node.
    /// Returns Ok(Some(FloatValue)) if it's an ExpressionStmt, Ok(None) for LetBinding, Err on failure.
    /// Compiles a single Statement node. Now runs FPM on function definitions.
    fn compile_statement(
        &mut self,
        stmt: &Statement,
    ) -> CompileResult<'ctx, Option<FloatValue<'ctx>>> {
        match stmt {
            Statement::LetBinding { name, value } => {
                // Compile the value expression first
                let compiled_value = self.compile_expression(value)?;

                // Check if variable already exists
                let alloca = if let Some(existing_alloca) = self.variables.get(name) {
                    *existing_alloca
                } else {
                    // Allocate memory for the new variable
                    let new_alloca = self.create_entry_block_alloca(name);
                    self.variables.insert(name.clone(), new_alloca);
                    new_alloca
                };

                // Store the compiled value into the allocated memory
                self.builder.build_store(alloca, compiled_value);

                Ok(None)
            }

            Statement::ExpressionStmt(expr) => {
                // Compile the expression
                let expr_value = self.compile_expression(expr)?;
                Ok(Some(expr_value))
            }

            Statement::FunctionDef { name, params, body } => {
                // Check for redefinition
                if self.functions.contains_key(name) {
                    return Err(CodeGenError::FunctionRedefinition(name.clone()));
                }

                // Define Function Type
                let f64_type = self.context.f64_type();
                let param_types: Vec<BasicMetadataTypeEnum> = std::iter::repeat(f64_type.into())
                    .take(params.len())
                    .collect();
                let fn_type = f64_type.fn_type(&param_types, false);

                // Create LLVM Function
                let function = self.module.add_function(name, fn_type, None);

                // Store Function Value before compiling body for possible recursion
                self.functions.insert(name.clone(), function);

                // Setup Function Body Context
                let entry_block = self.context.append_basic_block(function, "entry");
                let original_builder_pos = self.builder.get_insert_block();
                let original_func = self.current_function;
                let original_vars = self.variables.clone();

                self.builder.position_at_end(entry_block);
                self.current_function = Some(function);
                self.variables.clear();

                // Allocate and Store Parameters
                for (i, param_name) in params.iter().enumerate() {
                    let llvm_param = function.get_nth_param(i as u32).unwrap();
                    llvm_param.set_name(param_name);
                    let param_alloca = self.create_entry_block_alloca(param_name);
                    self.builder.build_store(param_alloca, llvm_param);
                    self.variables.insert(param_name.clone(), param_alloca);
                }

                // Compile Function Body Statements
                let mut last_body_val: Option<FloatValue<'ctx>> = None;
                let mut body_compile_err: Option<CodeGenError> = None;

                for body_stmt in &body.statements {
                    match self.compile_statement(body_stmt) {
                        Ok(Some(val)) => last_body_val = Some(val),
                        Ok(None) => {}
                        Err(e) => {
                            body_compile_err = Some(e);
                            break;
                        }
                    }
                }

                // Build Return
                if body_compile_err.is_none() {
                    if let Some(ret_val) = last_body_val {
                        self.builder.build_return(Some(&ret_val));
                    } else {
                        // No expression statement, return default 0.0
                        self.builder.build_return(Some(&f64_type.const_float(0.0)));
                    }
                }

                // Handle Body Compilation Error (before running passes)
                if let Some(err) = body_compile_err {
                    // Restore context before deleting
                    self.variables = original_vars;
                    self.current_function = original_func;
                    if let Some(bb) = original_builder_pos {
                        self.builder.position_at_end(bb);
                    }

                    self.functions.remove(name);
                    unsafe {
                        function.delete();
                    }
                    return Err(err);
                }

                // Verify before optimizing
                if !function.verify(true) {
                    eprintln!(
                        "Invalid function generated '{}' BEFORE optimization:\n{}",
                        name,
                        function.print_to_string().to_string()
                    );

                    // Restore context
                    self.variables = original_vars;
                    self.current_function = original_func;
                    if let Some(bb) = original_builder_pos {
                        self.builder.position_at_end(bb);
                    }

                    self.functions.remove(name);
                    unsafe {
                        function.delete();
                    }
                    return Err(CodeGenError::LlvmError(format!(
                        "Function '{}' verification failed before optimization",
                        name
                    )));
                }

                // Run the registered passes on the generated function
                let changed = self.fpm.run_on(&function);
                if changed {
                    println!("Optimizer changed function '{}'", name);
                }

                // Restore Outer Context (AFTER passes)
                self.variables = original_vars;
                self.current_function = original_func;
                if let Some(original_block) = original_builder_pos {
                    self.builder.position_at_end(original_block);
                }

                // Re-verify after optimization
                if function.verify(true) {
                    Ok(None) // Function definition successful
                } else {
                    eprintln!(
                        "Invalid function generated '{}' AFTER optimization:\n{}",
                        name,
                        function.print_to_string().to_string()
                    );
                    self.functions.remove(name);
                    unsafe {
                        function.delete();
                    }
                    return Err(CodeGenError::LlvmError(format!(
                        "Function '{}' verification failed after optimization",
                        name
                    )));
                }
            }
        }
    }

    /// Compile the program into the module, generating a standard C main function
    /// that calls printf with the result of the last top-level expression.
    pub fn compile_program_to_module(&mut self, program: &Program) -> Result<(), CodeGenError> {
        // --- Reset state for this compilation unit ---
        self.functions.clear(); // Clear user function definitions (printf is handled separately)

        // --- Declare Printf Early ---
        // Ensure printf is declared before compiling user code that might need it (though not used directly by user code yet)

        // --- Define User Functions ---
        // First pass: Define all user functions so they are available for calls
        // Need to be careful about compile_statement potentially modifying state needed later
        // Let's compile functions within the main loop for now, requires functions defined before use.
        // A multi-pass approach might be better later.

        // --- Setup C-Style Main Function ---
        let i32_type = self.context.i32_type();
        // Standard C main signature: int main() (or int main(int argc, char** argv))
        // We'll use the simple int main() for now.
        let main_fn_type = i32_type.fn_type(&[], false);
        let main_function = self.module.add_function(
            "main", // Standard C entry point name
            main_fn_type,
            None, // Default linkage (usually external)
        );
        let entry_block = self.context.append_basic_block(main_function, "entry");

        // --- Save/Restore context ---
        let original_builder_pos = self.builder.get_insert_block();
        let original_func = self.current_function;
        let original_vars = self.variables.clone();
        self.builder.position_at_end(entry_block);
        self.current_function = Some(main_function);
        self.variables.clear();

        // --- Compile Top-Level Statements ---
        // let mut last_main_expr_value: Option<FloatValue<'ctx>> = None;
        for stmt in &program.statements {
            self.compile_statement(stmt)?;
            // match self.compile_statement(stmt)? {
            //     Some(value) => last_main_expr_value = Some(value),
            //     None => {} // LetBinding or FunctionDef
            // }
        }

        // --- Build Return for C Main (return 0) ---
        self.builder
            .build_return(Some(&i32_type.const_int(0, false))); // return 0;

        // --- Restore Compiler State ---
        // ... (restore variables, current_function, builder position) ...
        self.variables = original_vars;
        self.current_function = original_func;
        if let Some(bb) = original_builder_pos {
            self.builder.position_at_end(bb);
        }

        // --- Optimize and Verify Main ---
        // No need to optimize here if we optimize later or let TargetMachine handle it?
        // Let's keep the FPM run on main for consistency for now.
        if main_function.verify(true) {
            let changed = self.fpm.run_on(&main_function);
            if changed {
                println!("Optimizer changed function 'main'");
            }
            if !main_function.verify(true) {
                // Handle verification failure after optimization
                // unsafe { main_function.delete(); } // No need to delete, just return Err
                return Err(CodeGenError::LlvmError(
                    "Main function verification failed after optimization".to_string(),
                ));
            }
        } else {
            // Handle verification failure before optimization
            // unsafe { main_function.delete(); }
            return Err(CodeGenError::LlvmError(
                "Main function verification failed before optimization".to_string(),
            ));
        }

        Ok(()) // Module is ready
    }

    /// Emits the compiled module to an object file.
    /// Should be called *after* compile_program_to_module.
    pub fn emit_object_file(&self, output_path: &Path) -> Result<(), CodeGenError> {
        // --- Target Configuration ---
        // Determine Target Triple for M4 Pro Mac
        let target_triple = &TargetTriple::create("aarch64-apple-darwin"); // ARM64 macOS

        // Initialize required targets (native is usually sufficient for host compilation)
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| {
            CodeGenError::LlvmError(format!("Failed to initialize native target: {}", e))
        })?;
        // Alternatively, initialize specific targets: Target::initialize_aarch64(&InitializationConfig::default());

        // --- Target Lookup ---
        let target = Target::from_triple(target_triple).map_err(|e| {
            CodeGenError::LlvmError(format!(
                "Failed to get target for triple '{}': {}",
                target_triple, e
            ))
        })?;

        // --- Create Target Machine ---
        // Configure CPU, features, optimization level etc.
        let cpu = "generic"; // Be specific for M4 - "generic" or "" might also work
        let features = ""; // Use "" for default features for the CPU
        let opt_level = OptimizationLevel::Default; // Or match FPM level used
        let reloc_mode = RelocMode::Default; // Position Independent Code (PIC) is common default
        let code_model = CodeModel::Default; // Small/Kernel/Medium/Large - Default is usually fine

        let target_machine = target
            .create_target_machine(
                target_triple,
                cpu,
                features,
                opt_level,
                reloc_mode,
                code_model,
            )
            .ok_or_else(|| {
                CodeGenError::LlvmError(format!(
                    "Failed to create target machine for triple '{}'",
                    target_triple
                ))
            })?;

        // --- Emit Object File ---
        println!("Emitting object file to: {}", output_path.display());
        target_machine
            .write_to_file(
                &self.module,     // The LLVM module containing the compiled code
                FileType::Object, // Specify we want an object file
                output_path,
            )
            .map_err(|e| CodeGenError::LlvmError(format!("Failed to write object file: {}", e)))?;

        Ok(())
    }
}

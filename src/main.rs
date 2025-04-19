// src/main.rs

mod token;
mod lexer;
mod ast;
mod parser;
mod codegen;

use lexer::Lexer;
use parser::Parser;
use codegen::Compiler;

// Import inkwell types needed for JIT
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction}; // Added JitFunction
use inkwell::OptimizationLevel; // To specify optimization level for JIT

// Define the function signature for our JITed 'main' function
// It takes no arguments and returns an f64 (double)
type MainFuncSignature = unsafe extern "C" fn() -> f64;

fn main() {
    let input = "1 + 2.5 * ( 30 / 4 - 2) - .5";
    // Try: "5.0 * 3.0" -> 15.0
    // Try: "100.0 / (2.0 * 5.0)" -> 10.0

    println!("Input: {}", input);

    // --- Parsing ---
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let ast = match parser.parse_expression() {
        Ok(ast) => {
            println!("AST:\n{:#?}", ast);
            ast
        }
        Err(e) => {
            eprintln!("Parsing Error: {}", e);
            return;
        }
    };

    // --- Code Generation & JIT ---
    let context = Context::create();
    let module = context.create_module("toy_jit"); // Module name can be anything
    let builder = context.create_builder();

    // Compiler setup
    let mut compiler = Compiler::new(&context, &builder, &module);

    // Compile the AST
    let main_function = match compiler.compile(&ast) {
        Ok(f) => {
            println!("\n--- Generated LLVM IR ---");
            f.print_to_stderr();
            f // Return the function on success
        }
        Err(e) => {
            eprintln!("\nCode Generation Error: {}", e);
            module.print_to_stderr(); // Print module IR even on error for debugging
            return;
        }
    };

    // --- JIT Execution ---
    println!("\n--- JIT Execution ---");

    // 1. Initialize LLVM target components needed for JIT
    //    We initialize for the native target.
    inkwell::targets::Target::initialize_native(&inkwell::targets::InitializationConfig::default())
        .expect("Failed to initialize native target");

    // 2. Create the JIT Execution Engine
    //    We need a Result here as JIT creation can fail (e.g., if the target isn't supported)
    let execution_engine_result: Result<ExecutionEngine, _> =
        module.create_jit_execution_engine(OptimizationLevel::None); // Use OptLevel::None for now, see Note below

    let execution_engine = match execution_engine_result {
        Ok(ee) => ee,
        Err(err) => {
            eprintln!("Failed to create Execution Engine: {}", err);
            return;
        }
    };

    // 3. Get a callable function pointer from the JIT engine
    //    The function name must match the one we added in codegen (`"main"`)
    let main_func_jit_result: Result<JitFunction<MainFuncSignature>, _> =
        unsafe { execution_engine.get_function("main") }; // unsafe: retrieving function ptrs

    let main_func_jit = match main_func_jit_result {
        Ok(f) => f,
        Err(err) => {
            eprintln!("Failed to get JIT function 'main': {}", err);
            // Often useful to dump the module IR if the function isn't found
            eprintln!("Module IR state:");
            module.print_to_stderr();
            return;
        }
    };


    // 4. Call the JIT-compiled function
    println!("Executing JITed code...");
    let result = unsafe { main_func_jit.call() }; // unsafe: calling raw machine code
    println!("JIT Execution Result: {}", result);

} // Context, Module, Builder, ExecutionEngine are dropped here, cleaning up LLVM resources.
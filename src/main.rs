// src/main.rs

mod ast;
mod codegen;
mod lexer;
mod parser;
mod token;

// Keep Parser, call parse_program
// Use Program codegen entry point
use codegen::Compiler;
use lexer::Lexer;
// Use Program parser entry point
use parser::Parser;
// Keep Compiler, call compile_program
// src/main.rs

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
// ... imports ...
use inkwell::OptimizationLevel;

type MainFuncSignature = unsafe extern "C" fn() -> f64;

fn main() {
    let input = r#"
        fun multiply(a, b) {
            let result = a * b; // Define local var in func
            result;             // Last expr is return value
        }

        let x = 10.0;
        let y = 5.0;
        multiply(x, y + 2.0); // Call the function: 10.0 * (5.0 + 2.0) = 70.0
    "#;
    // Try: "fun id(x) { x; } id(123);" -> 123.0
    // Try: "fun adder(a,b,c) {a+b+c;} adder(1,2,3);" -> 6.0

    println!("Input:\n{}", input);

    // --- Parsing ---
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let program = match parser.parse_program() {
        Ok(prog) => {
            println!("AST:\n{:#?}", prog);
            prog
        }
        Err(errors) => {
            eprintln!("Parsing Errors:");
            for e in errors {
                eprintln!("- {}", e);
            }
            return;
        }
    };

    // --- Code Generation & JIT ---
    let context = Context::create();
    let module = context.create_module("toy_jit_func");
    let builder = context.create_builder();
    let mut compiler = Compiler::new(&context, &builder, &module);

    // Compile the program. This generates 'main' and also compiles function definitions found.
    let main_function = match compiler.compile_program(&program) {
        Ok(f) => {
            println!("\n--- Generated LLVM IR (Module) ---");
            // Print the whole module now to see both functions
            module.print_to_stderr();
            f // Return the main function value
        }
        Err(e) => {
            eprintln!("\nCode Generation Error: {}", e);
            eprintln!("--- Module IR State on Error ---");
            module.print_to_stderr();
            return;
        }
    };

    // --- JIT Execution --- (remains the same)
    // ... initialize_native, create_jit_execution_engine ...
    // ... get_function("main"), call ...
    println!("\n--- JIT Execution ---");
    inkwell::targets::Target::initialize_native(&inkwell::targets::InitializationConfig::default())
        .expect("Failed to initialize native target");
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .expect("Failed to create Execution Engine");
    let main_func_jit: JitFunction<MainFuncSignature> =
        unsafe { execution_engine.get_function("main") }
            .expect("Failed to find JIT function 'main'");

    println!("Executing JITed code...");
    let result = unsafe { main_func_jit.call() };
    println!("JIT Execution Result: {}", result);
}

// src/main.rs

mod ast;
mod codegen;
mod lexer;
mod parser;
mod token;

use lexer::Lexer;
// Use Program parser entry point
use parser::Parser;
// Keep Parser, call parse_program
// Use Program codegen entry point
use codegen::Compiler;
// Keep Compiler, call compile_program

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::OptimizationLevel;

type MainFuncSignature = unsafe extern "C" fn() -> f64;

fn main() {
    let input = r#"
        let x = 10.0;
        let y = x + 5.0;
        let factor = 2.0;
        y * factor;
    "#;// This should be the return value: (10+5)*2 = 30.0
    // Try: "let a = 1; let b = 2; a+b;" -> 3.0
    // Try: "5.0 * (2.0 + 3.0);" -> 25.0
    // Try: "let z=100;" -> Returns 0.0 as last statement isn't expression

    println!("Input:\n{}", input);

    // --- Parsing ---
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let program = match parser.parse_program() {
        // Call parse_program
        Ok(prog) => {
            println!("AST:\n{:#?}", prog);
            prog
        }
        Err(errors) => {
            // Handle multiple errors
            eprintln!("Parsing Errors:");
            for e in errors {
                eprintln!("- {}", e);
            }
            return;
        }
    };

    // --- Code Generation & JIT ---
    let context = Context::create();
    let module = context.create_module("toy_jit_stmt");
    let builder = context.create_builder();
    let mut compiler = Compiler::new(&context, &builder, &module);

    let main_function = match compiler.compile_program(&program) {
        // Call compile_program
        Ok(f) => {
            println!("\n--- Generated LLVM IR ---");
            f.print_to_stderr();
            f
        }
        Err(e) => {
            eprintln!("\nCode Generation Error: {}", e);
            eprintln!("--- Module IR State on Error ---");
            module.print_to_stderr();
            return;
        }
    };

    // --- JIT Execution --- (remains the same)
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

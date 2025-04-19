// src/main.rs

mod ast;
mod codegen;
mod lexer;
mod parser;
mod token;

use codegen::Compiler;
use lexer::Lexer;
use parser::Parser;
// src/main.rs

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
// ... imports ...
use inkwell::OptimizationLevel;

type MainFuncSignature = unsafe extern "C" fn() -> f64;

fn main() {
    // Input with variables
    let input = "let x = 10 in let y = x + 5 in y * 2"; // Expected: (10 + 5) * 2 = 30.0
    // Try: "let a=1 in let b=2 in a+b" -> 3.0
    // Try: "let factor = 5.0 in factor * (factor - 2.0)" -> 5.0 * 3.0 = 15.0
    // Try: "let x=1 in x + (let x=2 in x)" -> Should this be 1 + 2 = 3 (shadowing)? Yes, our current logic handles this.

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
            // Maybe print tokens seen so far if debugging lexer/parser interaction
            return;
        }
    };

    // --- Code Generation & JIT ---
    let context = Context::create();
    let module = context.create_module("toy_jit_var");
    let builder = context.create_builder();
    let mut compiler = Compiler::new(&context, &builder, &module);

    let main_function = match compiler.compile(&ast) {
        Ok(f) => {
            println!("\n--- Generated LLVM IR ---");
            f.print_to_stderr();
            f
        }
        Err(e) => {
            eprintln!("\nCode Generation Error: {}", e);
            // Dump module even on error to see partial IR
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
        .expect("Failed to create Execution Engine"); // Simplified error handling

    let main_func_jit: JitFunction<MainFuncSignature> =
        unsafe { execution_engine.get_function("main") }
            .expect("Failed to find JIT function 'main'"); // Simplified error handling

    println!("Executing JITed code...");
    let result = unsafe { main_func_jit.call() };
    println!("JIT Execution Result: {}", result);
}

// src/main.rs

mod ast;
mod codegen;
mod lexer;
mod parser;
mod token; // Add codegen module

use codegen::Compiler;
use lexer::Lexer;
use parser::Parser;
// Import the Compiler

// Import inkwell types needed in main
use inkwell::context::Context;

fn main() {
    let input = "1 + 2.5 * ( 30 / 4 - 2) - .5";
    // Try other inputs: "10", "1+2+3", "5.0 * (2.0 + 3.0)"

    println!("Input: {}", input);

    // --- Parsing ---
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let ast = match parser.parse_expression() {
        Ok(ast) => {
            println!("AST:\n{:#?}", ast);
            ast // Return the AST
        }
        Err(e) => {
            eprintln!("Parsing Error: {}", e);
            return; // Exit if parsing failed
        }
    };

    // --- Code Generation ---
    let context = Context::create(); // Create the single LLVM context
    let module = context.create_module("toy_module"); // Create a module named "toy_module"
    let builder = context.create_builder(); // Create a builder

    // Create the Compiler instance
    let mut compiler = Compiler::new(&context, &builder, &module);

    // Compile the AST into an LLVM function
    match compiler.compile(&ast) {
        Ok(main_function) => {
            println!("\n--- Generated LLVM IR ---");
            // Print the IR to the console
            main_function.print_to_stderr(); // Or print_to_string()

            // You can also dump the entire module's IR
            println!("\n--- Module IR ---");
            module.print_to_stderr();

            // --- Verification (Optional here, as compile() already does it) ---
            // if !main_function.verify(true) {
            //     eprintln!("\nLLVM Function Verification Failed after compile!");
            //     // module.print_to_stderr(); // Dump module IR for inspection
            // } else {
            //     println!("\nLLVM Function Verification Succeeded.");
            // }

            // --- Next Steps (Future Chapters) ---
            // Optimization: module.run_passes(...)
            // JIT Execution: module.create_jit_execution_engine(...)
            // Compilation to Object File: module.write_bitcode_to_path(...) or TargetMachine setup
        }
        Err(e) => {
            eprintln!("\nCode Generation Error: {}", e);
            // Optionally print partial IR if module exists
            // module.print_to_stderr();
        }
    }
}

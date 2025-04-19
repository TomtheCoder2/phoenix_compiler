// src/main.rs

mod ast;
mod codegen;
mod lexer;
mod parser;
mod token;

use codegen::Compiler;
use lexer::Lexer;
use parser::Parser;

use inkwell::context::Context;
use std::path::Path;
// For output file path
use std::process::Command;
// To call linker (e.g., clang)

// Remove JIT type alias if not used
// type MainFuncSignature = unsafe extern "C" fn() -> f64;

fn main() {
    let input = r#"
        fun multiply(a, b) {
            let result = a * b;
            result;
        }

        fun calculate(a, b) {
            let x = a + 1.0;
            let y = b + 2.0;
            let z = x * y;
            z - a;
        }

        let x = calculate(5.0, 10.0); // 67.0
        multiply(x, 2.0);            // Expected result: 67.0 * 2.0 = 134.0
    "#;

    let output_filename = "output.o"; // Name for the object file

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

    // --- Code Generation ---
    let context = Context::create();
    let module = context.create_module("toy_module_obj"); // Module name
    let builder = context.create_builder();
    let mut compiler = Compiler::new(&context, &builder, &module);

    // Compile the program AST into the LLVM module
    match compiler.compile_program_to_module(&program) {
        // Use new method
        Ok(()) => {
            println!("\n--- LLVM IR (Module) ---");
            module.print_to_stderr(); // Print IR before emission
        }
        Err(e) => {
            eprintln!("\nCode Generation Error: {}", e);
            eprintln!("--- Module IR State on Error ---");
            module.print_to_stderr();
            return;
        }
    };

    // --- Emit Object File ---
    let obj_path = Path::new(output_filename);
    match compiler.emit_object_file(obj_path) {
        Ok(()) => {
            println!("Successfully emitted object file: {}", output_filename);
        }
        Err(e) => {
            eprintln!("\nObject File Emission Error: {}", e);
            return;
        }
    }

    // --- Linking (Optional - using external linker like clang) ---
    link_object_file(obj_path, "output_executable"); // Specify desired executable name
} // Context, Module, Builder dropped here

/// Links the generated object file using an external linker (e.g., clang).
/// Assumes a 'main' function compatible with C linking exists in the object file.
/// NOTE: Our current 'main' returns f64, which isn't the standard C main signature (int()).
/// Linking might succeed, but running it might crash or behave unexpectedly
/// without a proper C entry point or runtime setup.
/// For now, we just demonstrate the linking command.
fn link_object_file(obj_path: &Path, executable_name: &str) {
    println!("\n--- Linking ---");
    let linker = "clang"; // Use clang as linker (handles C library linking)
    println!("Attempting to link {} using {}", obj_path.display(), linker);

    let status = Command::new(linker)
        .arg(obj_path) // Input object file
        .arg("-o") // Specify output executable name
        .arg(executable_name)
        // Add libraries if needed, e.g. .arg("-lm") for math library
        .status(); // Execute the command

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("Successfully linked executable: {}", executable_name);
            println!("You can inspect it with: file {}", executable_name);
            // To run (might crash/print garbage): ./{executable_name}; echo $?
        }
        Ok(exit_status) => {
            eprintln!("Linking failed with status: {}", exit_status);
        }
        Err(e) => {
            eprintln!("Failed to execute linker '{}': {}", linker, e);
            eprintln!("Ensure '{}' is installed and in your PATH.", linker);
        }
    }
}

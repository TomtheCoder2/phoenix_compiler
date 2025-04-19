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
// To call linker (e.g., clang)
use std::io::{self, Write};
use std::path::Path;
// For output file path
use std::process::Command;
// Add import

// Remove JIT type alias if not used
// type MainFuncSignature = unsafe extern "C" fn() -> f64;

#[unsafe(no_mangle)]
pub extern "C" fn print_f64_wrapper(value: f64) {
    println!("{:.6}", value); // Or use libc::printf if you add the dependency
}

fn main() {
    let input = r#"
        print(5);
        print(6);
        print(7);
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
// src/main.rs -> link_object_file function

fn link_object_file(obj_path: &Path, executable_name: &str) {
    println!("\n--- Linking ---");
    let linker = "clang";
    // Determine the path to the static library produced by cargo build
    // Usually in target/debug/ or target/release/
    // Use CARGO_TARGET_DIR if set, otherwise assume ./target/debug
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    // Adjust debug/release based on your build profile
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let lib_name = "libtoylang_compiler.a"; // Matches package name by default
    let lib_path = Path::new(&target_dir).join(profile).join(lib_name);

    if !lib_path.exists() {
        eprintln!(
            "Linking Error: Static library not found at {}",
            lib_path.display()
        );
        eprintln!("Ensure you have run 'cargo build' first.");
        return;
    }

    println!(
        "Attempting to link {} with library {} using {}",
        obj_path.display(),
        lib_path.display(),
        linker
    );

    let status = Command::new(linker)
        .arg(obj_path) // Input object file from ToyLang code
        .arg(lib_path) // Input static library containing the wrapper
        .arg("-o")
        .arg(executable_name)
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("Successfully linked executable: {}", executable_name);
            println!("You should now be able to run './{}'", executable_name);
        }
        Ok(exit_status) => {
            eprintln!("Linking failed with status: {}", exit_status);
        }
        Err(e) => {
            eprintln!("Failed to execute linker '{}': {}", linker, e);
        }
    }
}

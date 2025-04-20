
use inkwell::context::Context;
// To call linker (e.g., clang)
use std::io::{self, Write};
use std::path::Path;
// For output file path
use std::process::Command;
use toylang_compiler::codegen::Compiler;
use toylang_compiler::lexer::Lexer;
use toylang_compiler::utils::link_object_file;
use toylang_compiler::parser::Parser;
// Add import

// Remove JIT type alias if not used
// type MainFuncSignature = unsafe extern "C" fn() -> f64;

fn main() {
    let input = r#"
        var counter: int = 0;
        let limit: int = 5;
        // let immutable: int = 100; // Example immutable

        print(counter); // 0

        counter = counter + 1;
        print(counter); // 1

        counter = counter + limit;
        print(counter); // 6

        // immutable = 50; // This should fail if uncommented (after adding check)

        let result = counter * 2; // Use mutable var in expression
        print(result); // 12
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



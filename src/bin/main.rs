use inkwell::context::Context;
use std::path::Path;
use phoenix_compiler::codegen::Compiler;
use phoenix_compiler::lexer::Lexer;
use phoenix_compiler::parser::Parser;
use phoenix_compiler::typechecker::TypeChecker;
use phoenix_compiler::utils::link_object_file;
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
    let lexer = Lexer::new("main.rs".to_string(), input);
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
    
    let mut type_checker = TypeChecker::new();
    // Type check the program
    let symbol_table = match type_checker.check_program(&program) {
        Ok(v) => {
            println!("Type Checking Successful.");
            v
        }
        Err(errors) => {
            eprintln!("Type Checking Errors:");
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
    let mut compiler = Compiler::new(&context, &builder, &module, symbol_table);

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

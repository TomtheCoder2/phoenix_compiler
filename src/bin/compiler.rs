use clap::Parser;
use inkwell::context::Context;
use std::fs;
use std::path::Path;
use phoenix_compiler::codegen::Compiler;
use phoenix_compiler::lexer::Lexer;
use phoenix_compiler::parser::Parser as ToyParser;
use phoenix_compiler::typechecker::TypeChecker;
use phoenix_compiler::utils::link_object_file;

#[derive(Parser, Debug)]
#[command(author, version, about = "phoenix Compiler")]
struct Args {
    /// Input file containing phoenix code
    #[arg(short, long)]
    input: String,

    /// Output executable file path
    #[arg(short, long, default_value = "output/output")]
    output: String,

    /// Enable verbose output
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    // Parse command line arguments
    let args = Args::parse();

    // Read input file
    let input_path = Path::new(&args.input);
    let output_path = Path::new(&args.output);
    // check if the output directory exists
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            eprintln!("Output directory does not exist: {}", parent.display());
            // create it
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("Error creating output directory: {}", e);
                return;
            }
        }
    }
    let input = match fs::read_to_string(input_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading input file '{}': {}", args.input, e);
            return;
        }
    };

    if args.verbose {
        println!("Compiling file: {}", args.input);
    }

    // Parse the input code
    let lexer = Lexer::new(input_path.display().to_string(), &input);
    let mut parser = ToyParser::new(lexer);
    let program = match parser.parse_program() {
        Ok(prog) => {
            if args.verbose {
                println!("Parsing successful");
            }
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

    // write the ast to file if verbose
    if args.verbose {
        let ast_file_path = output_path.with_extension("ast");
        match fs::write(&ast_file_path, format!("{:#?}", program)) {
            Ok(_) => println!("AST written to: {}", ast_file_path.display()),
            Err(e) => eprintln!("Error writing AST to file: {}", e),
        }
    }

    println!("\n--- Type Checking ---");
    let mut type_checker = TypeChecker::new();
    match type_checker.check_program(&program) {
        // Call the checker
        Ok(()) => {
            println!("Type Checking Successful.");
        }
        Err(errors) => {
            eprintln!("Type Checking Errors:");
            for e in errors {
                eprintln!("- {}", e);
            }
            std::process::exit(1); // Exit if errors found
        }
    }

    // Generate LLVM IR
    let context = Context::create();
    let module = context.create_module("toy_module_obj");
    let builder = context.create_builder();
    let mut compiler = Compiler::new(&context, &builder, &module);

    // Compile the program AST to LLVM IR
    match compiler.compile_program_to_module(&program) {
        Ok(()) => {
            if args.verbose {
                println!("Code generation successful");
            }
        }
        Err(e) => {
            eprintln!("Code Generation Error: {}", e);
            return;
        }
    };

    // write the llvm ir to file if verbose
    if args.verbose {
        let llvm_ir_file_path = output_path.with_extension("ll");
        match fs::write(&llvm_ir_file_path, module.print_to_string().to_string()) {
            Ok(_) => println!("LLVM IR written to: {}", llvm_ir_file_path.display()),
            Err(e) => eprintln!("Error writing LLVM IR to file: {}", e),
        }
    }

    // Create a temporary object file path
    let temp_obj_path = Path::new("temp_output.o");

    // Emit the object file
    match compiler.emit_object_file(temp_obj_path) {
        Ok(()) => {
            if args.verbose {
                println!("Object file emission successful");
            }
        }
        Err(e) => {
            eprintln!("Object File Emission Error: {}", e);
            return;
        }
    }

    // Link the object file to create the executable
    link_object_file(temp_obj_path, &args.output);

    // Clean up the temporary object file
    if let Err(e) = fs::remove_file(temp_obj_path) {
        if args.verbose {
            eprintln!("Warning: Failed to remove temporary object file: {}", e);
        }
    }

    println!(
        "Compilation successful! Executable saved to: {}",
        args.output
    );
}

use clap::Parser;
use inkwell::context::Context;
use std::fs;
use std::path::Path;
use toylang_compiler::codegen::Compiler;
use toylang_compiler::lexer::Lexer;
use toylang_compiler::utils::link_object_file;
use toylang_compiler::parser::Parser as ToyParser;

#[derive(Parser, Debug)]
#[command(author, version, about = "ToyLang Compiler")]
struct Args {
    /// Input file containing ToyLang code
    #[arg(short, long)]
    input: String,

    /// Output executable file path
    #[arg(short, long)]
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
    let lexer = Lexer::new(&input);
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

    println!("Compilation successful! Executable saved to: {}", args.output);
}
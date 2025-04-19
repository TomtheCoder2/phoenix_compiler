// src/main.rs

mod token;
mod lexer;
mod ast; // Add the ast module
mod parser; // Add the parser module

use lexer::Lexer;
use parser::Parser; // Use the Parser

fn main() {
    // More complex input to test precedence and associativity
    let input = "2 * 1 + 2";

    println!("Input: {}", input);

    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);

    match parser.parse_expression() {
        Ok(ast) => {
            println!("AST:\n{:#?}", ast); // Print the formatted AST
        }
        Err(e) => {
            eprintln!("Parsing Error: {}", e); // Print the Display impl of ParseError
        }
    }
}
// src/lib.rs (Create this file)
use std::io::Write;
pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod token;
pub mod types;
pub mod utils;

#[no_mangle]
pub extern "C" fn print_f64_wrapper(value: f64) {
    println!("{:.6}", value);
}

#[no_mangle]
pub extern "C" fn print_i64_wrapper(value: i64) {
    println!("{}", value); // Rust's default i64 formatting
}

#[no_mangle]
pub extern "C" fn print_bool_wrapper(value: bool) {
    // LLVM i1 maps to bool in Rust FFI
    println!("{}", value); // Prints "true" or "false"
}

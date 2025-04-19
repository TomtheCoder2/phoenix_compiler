// src/lib.rs (Create this file)
use std::io::{self, Write};

#[no_mangle]
pub extern "C" fn print_f64_wrapper(value: f64) {
    println!("{:.6}", value);
}

// Add other runtime functions here later
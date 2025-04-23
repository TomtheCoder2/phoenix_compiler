// Add imports for runtime
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;
use crate::vector::PhoenixVec;

pub mod vector;
pub mod string;
// For pointer type

#[no_mangle]
pub extern "C" fn print_f64_wrapper(value: f64) {
    print!("{:.6}", value);
}

#[no_mangle]
pub extern "C" fn print_i64_wrapper(value: i64) {
    print!("{}", value); // Rust's default i64 formatting
}

#[no_mangle]
pub extern "C" fn print_bool_wrapper(value: bool) {
    // LLVM i1 maps to bool in Rust FFI
    print!("{}", value); // Prints "true" or "false"
}

#[no_mangle]
pub extern "C" fn print_str_wrapper(handle: *const libc::c_void) {
    // Takes handle now, not C char*
    if handle.is_null() {
        println!("<NULL_STR_HANDLE>");
        return;
    }
    let header = unsafe { &*(handle as *const PhoenixVec) };
    // Create a Rust string slice directly from the data pointer and length
    let byte_slice = unsafe { slice::from_raw_parts(header.data, header.length as usize) };
    match std::str::from_utf8(byte_slice) {
        Ok(rust_str) => print!("{}", rust_str), // Use println
        Err(_) => eprintln!("<INVALID_UTF8_STR>"),
    }
}

#[no_mangle]
pub extern "C" fn print_str_ln_wrapper_func() {
    println!(); // Just print a newline
}

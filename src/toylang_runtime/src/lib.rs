use std::ffi::CStr;
use std::os::raw::c_char;
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
pub unsafe extern "C" fn print_str_wrapper(c_string_ptr: *const c_char) {
    if c_string_ptr.is_null() {
        // Handle null pointer case if necessary
        println!("<NULL_STR>");
        return;
    }
    // Unsafe block needed to dereference raw C pointer
    let c_str = unsafe { CStr::from_ptr(c_string_ptr) };
    // Convert CStr to Rust String slice (&str)
    match c_str.to_str() {
        Ok(rust_str) => {
            // println!("string: {:?}", rust_str.chars().collect::<Vec<char>>());
            print!("{}", rust_str)
        } // Use println for Rust string
        Err(_) => eprintln!("<INVALID_UTF8_STR>"), // Handle potential UTF-8 error
    }
    // No need to flush usually, println! does.
}

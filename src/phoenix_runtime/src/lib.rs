// Add imports for runtime
use std::alloc::{alloc, dealloc, Layout};
// Add imports for runtime
use std::ffi::CStr;
use std::os::raw::c_char;
// For manual allocation if not using Box/Vec directly
use std::ptr;
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

// The internal representation of a phoenix vector handle
#[repr(C)] // Ensure predictable layout for C interop
struct phoenixVec {
    capacity: i64,  // Number of elements allocated
    length: i64,    // Number of elements currently stored
    elem_size: i64, // Size of each element in bytes
    data: *mut u8,  // Raw pointer to heap-allocated element data
}

// Helper function to get layout for data allocation
fn data_layout(elem_size: i64, capacity: i64) -> Option<Layout> {
    Layout::from_size_align((elem_size * capacity) as usize, elem_size as usize).ok()
    // Use elem_size for alignment, assuming elements need standard alignment
    // Might need adjustment for complex types later
}

// fn(elem_size: i64, capacity: i64) -> *mut phoenixVec (as *mut c_void / i8*)
#[no_mangle]
pub extern "C" fn _phoenix_vec_new(elem_size: i64, capacity: i64) -> *mut libc::c_void {
    if elem_size <= 0 || capacity < 0 {
        eprintln!("RUNTIME ERROR: Invalid size/capacity for vec_new");
        return ptr::null_mut(); // Return null on error
    }

    // Allocate data buffer on the heap
    let data_ptr = if capacity == 0 {
        ptr::null_mut() // No allocation needed for zero capacity
    } else {
        match data_layout(elem_size, capacity) {
            Some(layout) => unsafe { alloc(layout) },
            None => {
                eprintln!("RUNTIME ERROR: Failed to create memory layout for vector data");
                return ptr::null_mut();
            }
        }
    };

    if capacity > 0 && data_ptr.is_null() {
        eprintln!("RUNTIME ERROR: Failed to allocate memory for vector data");
        // No need to free layout here
        return ptr::null_mut(); // Allocation failed
    }

    // Allocate header struct on the heap using Box, then leak it to get a stable pointer
    let header = Box::new(phoenixVec {
        capacity,
        length: capacity, // We assume that the vector is initialized with the given capacity // todo maybe specify an argument initial length
        elem_size,
        data: data_ptr,
    });

    // Return the raw pointer to the header (leaked Box) as the handle
    Box::into_raw(header) as *mut libc::c_void
}

// fn(vec_handle: *mut c_void / i8*)
#[no_mangle]
pub extern "C" fn _phoenix_vec_free(handle: *mut libc::c_void) {
    if handle.is_null() {
        return;
    } // Do nothing if null

    unsafe {
        // Reconstruct the Box from the raw pointer TO TAKE OWNERSHIP BACK
        let header_box = Box::from_raw(handle as *mut phoenixVec);

        // Deallocate the data buffer if it was allocated
        if header_box.capacity > 0 && !header_box.data.is_null() {
            if let Some(layout) = data_layout(header_box.elem_size, header_box.capacity) {
                dealloc(header_box.data, layout);
            } else {
                // This shouldn't happen if layout was ok during alloc
                eprintln!("RUNTIME ERROR: Layout mismatch during vec_free data deallocation");
            }
        }
        // Header itself is freed when header_box goes out of scope here
    }
}

// fn(vec_handle: *mut c_void / i8*) -> i64
#[no_mangle]
pub extern "C" fn _phoenix_vec_len(handle: *mut libc::c_void) -> i64 {
    if handle.is_null() {
        return -1;
    } // Indicate error? Or 0? Let's use -1.
    let header = unsafe { &*(handle as *const phoenixVec) }; // Borrow header immutably
    header.length
}

// fn(vec_handle: *mut c_void / i8*, index: i64) -> *mut c_void / i8* (pointer to element)
#[no_mangle]
pub extern "C" fn _phoenix_vec_get_ptr(handle: *mut libc::c_void, index: i64) -> *mut libc::c_void {
    if handle.is_null() {
        eprintln!("RUNTIME ERROR: Null handle passed to vec_get_ptr");
        return ptr::null_mut();
    }
    let header = unsafe { &*(handle as *const phoenixVec) }; // Borrow header

    // --- Bounds Check --- (Essential!)
    if index < 0 || index >= header.length {
        // Using eprintln for runtime errors visible during testing
        eprintln!(
            "RUNTIME ERROR: Index {} out of bounds for vector length {}",
            index, header.length
        );
        // What to do on error? Return null? Panic? Returning null is safer for C interop.
        return ptr::null_mut();
    }

    // Calculate pointer (careful with types/casting)
    let offset = index * header.elem_size;
    unsafe { header.data.offset(offset as isize) as *mut libc::c_void }
}

// fn(vec_handle: *mut c_void / i8*, value_ptr: *const c_void / i8*) -> void
#[no_mangle]
pub extern "C" fn _phoenix_vec_push(handle: *mut libc::c_void, value_ptr: *const libc::c_void) {
    if handle.is_null() || value_ptr.is_null() {
        eprintln!("RUNTIME ERROR: Null handle or value passed to vec_push");
        return;
    }

    let header = unsafe { &mut *(handle as *mut phoenixVec) }; // Need mutable header

    // --- Resize / Reallocate if needed ---
    if header.length >= header.capacity {
        let new_capacity = if header.capacity == 0 {
            4
        } else {
            header.capacity * 2
        }; // Double capacity
           // println!("Reallocating vector from {} to {}", header.capacity, new_capacity); // Debug print

        let Some(new_layout) = data_layout(header.elem_size, new_capacity) else {
            eprintln!("RUNTIME ERROR: Failed to create layout for resize");
            return;
        };
        let Some(old_layout) = data_layout(header.elem_size, header.capacity) else {
            eprintln!("RUNTIME ERROR: Failed to create old layout for resize");
            return; // Should not happen
        };

        let new_data_ptr = unsafe {
            if header.capacity == 0 {
                // Old pointer was null, just allocate new
                alloc(new_layout)
            } else {
                // Reallocate existing data
                std::alloc::realloc(header.data, old_layout, new_layout.size())
            }
        };

        if new_data_ptr.is_null() {
            eprintln!("RUNTIME ERROR: Vector resize reallocation failed");
            return; // Failed to grow
        }

        header.data = new_data_ptr;
        header.capacity = new_capacity;
    }

    // --- Copy Value to End ---
    // Calculate address of the next empty slot
    let dest_ptr = unsafe {
        header
            .data
            .offset((header.length * header.elem_size) as isize)
    };
    // Copy bytes from value_ptr to dest_ptr
    unsafe {
        ptr::copy_nonoverlapping(value_ptr as *const u8, dest_ptr, header.elem_size as usize);
    }

    // Increment length
    header.length += 1;
}

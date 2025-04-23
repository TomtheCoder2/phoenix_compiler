use std::ffi::CStr;
use std::os::raw::c_char;
use std::{ptr, slice};
use crate::vector::{PhoenixVec, _phoenix_vec_free, _phoenix_vec_get_ptr, _phoenix_vec_len, _phoenix_vec_new};

// fn(vec_handle: *mut c_void) -> *mut c_void (pointer to popped element, NULL if empty)
// Caller is responsible for knowing the type/size to copy from the returned pointer.
#[no_mangle]
pub extern "C" fn _phoenix_vec_pop(handle: *mut libc::c_void) -> *mut libc::c_void {
    if handle.is_null() {
        eprintln!("RUNTIME ERROR: Null handle passed to vec_pop");
        return ptr::null_mut();
    }
    let header = unsafe { &mut *(handle as *mut PhoenixVec) }; // Need mutable

    if header.length == 0 {
        eprintln!("RUNTIME ERROR: Pop from empty vector");
        return ptr::null_mut(); // Indicate empty or error
    }

    // Decrement length *first*
    header.length -= 1;
    // Calculate pointer to the (now technically previous) last element
    let offset = header.length * header.elem_size; // length is now index of last valid item + 1
    let elem_ptr = unsafe { header.data.offset(offset as isize) };

    // Return pointer to the location where the popped element *was*.
    // The caller needs to copy the data out *before* the next push potentially overwrites it.
    // NOTE: This doesn't actually remove/free the element data itself from the buffer,
    //       just makes the space available for the next push.
    elem_ptr as *mut libc::c_void
}

// --- String Runtime ---
// Very basic string handle - just a pointer to heap bytes + length?
// Or reuse VecHeader with elem_size=1? Let's use VecHeader for similarity.
// Using VecHeader means strings are internally Vec<u8> + metadata.

// fn(initial_content: *const c_char) -> *mut PhoenixVec (as *mut c_void)
#[no_mangle]
pub extern "C" fn _phoenix_str_new(content_ptr: *const c_char) -> *mut libc::c_void {
    let c_str = if content_ptr.is_null() {
        // Handle null input C string if necessary
        unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") } // Empty string
    } else {
        unsafe { CStr::from_ptr(content_ptr) }
    };
    let bytes = c_str.to_bytes(); // Get byte slice WITHOUT null terminator
    let len = bytes.len() as i64;

    // Allocate header + data for the string bytes (elem_size = 1)
    // Capacity could be len, or slightly larger for potential concatenation later
    let capacity = len; // Start with exact capacity
    let elem_size: i64 = 1;

    // Use vec_new logic essentially, but with elem_size=1 and initial content
    let handle = _phoenix_vec_new(elem_size, capacity);
    if handle.is_null() {
        return ptr::null_mut();
    }

    let header = unsafe { &mut *(handle as *mut PhoenixVec) };

    // Copy initial content into the data buffer
    if len > 0 {
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), header.data, len as usize);
        }
    }
    header.length = len; // Set the initial length

    handle // Return the handle
}

// fn(str_handle: *mut c_void) -> i64
#[no_mangle]
pub extern "C" fn _phoenix_str_len(handle: *mut libc::c_void) -> i64 {
    // Reuses vec_len logic as PhoenixVec stores length
    _phoenix_vec_len(handle)
}

// fn(str_handle: *mut c_void)
#[no_mangle]
pub extern "C" fn _phoenix_str_free(handle: *mut libc::c_void) {
    // Reuses vec_free as PhoenixVec holds the data pointer and metadata
    _phoenix_vec_free(handle)
}

// fn(str_handle: *mut c_void, index: i64) -> *mut c_void / i8* (pointer to char)
#[no_mangle]
pub extern "C" fn _phoenix_str_get_char_ptr(
    handle: *mut libc::c_void,
    index: i64,
) -> *mut libc::c_void {
    // Reuses vec_get_ptr as elem_size is 1
    _phoenix_vec_get_ptr(handle, index)
}

// String concatenation - more complex, involves creating a *new* string
// fn(left_handle: *mut c_void, right_handle: *mut c_void) -> *mut c_void
#[no_mangle]
pub extern "C" fn _phoenix_str_concat(
    left_handle: *mut libc::c_void,
    right_handle: *mut libc::c_void,
) -> *mut libc::c_void {
    if left_handle.is_null() || right_handle.is_null() {
        return ptr::null_mut();
    }
    let left_header = unsafe { &*(left_handle as *const PhoenixVec) };
    let right_header = unsafe { &*(right_handle as *const PhoenixVec) };

    let new_len = left_header.length + right_header.length;
    let new_capacity = new_len; // Create exact size for now
    let elem_size: i64 = 1;

    // Create new vector handle for the result
    let result_handle = _phoenix_vec_new(elem_size, new_capacity);
    if result_handle.is_null() {
        return ptr::null_mut();
    }
    let result_header = unsafe { &mut *(result_handle as *mut PhoenixVec) };

    // Copy left string bytes
    if left_header.length > 0 {
        unsafe {
            ptr::copy_nonoverlapping(
                left_header.data,
                result_header.data,
                left_header.length as usize,
            );
        }
    }
    // Copy right string bytes after left string bytes
    if right_header.length > 0 {
        let dest_ptr = unsafe { result_header.data.offset(left_header.length as isize) };
        unsafe {
            ptr::copy_nonoverlapping(right_header.data, dest_ptr, right_header.length as usize);
        }
    }

    result_header.length = new_len; // Set length of new string
    result_handle
}



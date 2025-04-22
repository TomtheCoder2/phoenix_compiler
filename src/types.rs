// src/types.rs

use inkwell::context::Context;
use inkwell::types::{AnyTypeEnum, BasicTypeEnum, FloatType, IntType};
// Import LLVM types
use std::fmt;
use inkwell::AddressSpace;

#[derive(Debug, Clone, PartialEq, Eq, Hash)] // Use Eq and Hash for HashMap keys if needed
pub enum Type {
    Float, // Represents f64
    Int,   // Represents i64
    Bool,  // Represents i1 (boolean)
    Void,  // Represents lack of a value (for functions that don't return) - maybe add later
    // Add Function, Array, Struct types later
    String,
    Vector(Box<Type>), // Added: Stores the element type, Boxed
}

impl Type {
    /// Get the corresponding LLVM BasicTypeEnum
    /// Panics if called on Type::Void or other non-basic types later.
    pub fn to_llvm_basic_type<'ctx>(&self, context: &'ctx Context) -> Option<BasicTypeEnum<'ctx>> {
        match self {
            Type::Float => Some(context.f64_type().into()),
            Type::Int => Some(context.i64_type().into()),
            Type::Bool => Some(context.bool_type().into()),
            // String represented as i8* (pointer is basic)
            Type::String => Some(context.ptr_type(AddressSpace::default()).into()),
            // Vector type maps to a pointer to our runtime struct (e.g., VecHeader*)
            // For now, let's represent it as a generic pointer (void*) or opaque struct pointer.
            // Need to define the runtime struct layout later.
            // Let's use i8* as a placeholder for the vector handle pointer.
            Type::Vector(_) => Some(context.ptr_type(AddressSpace::default()).into()),
            Type::Void => None, // Void is not a basic type
        }
    }

    /// Get the corresponding LLVM AnyTypeEnum (includes void, functions etc.)
    pub fn to_llvm_any_type<'ctx>(&self, context: &'ctx Context) -> AnyTypeEnum<'ctx> {
        match self {
            Type::Float => context.f64_type().into(),
            Type::Int => context.i64_type().into(),
            Type::Bool => context.bool_type().into(),
            Type::Void => context.void_type().into(),
            _ => panic!("Unhandled type in to_llvm_any_type: {:?}", self)
        }
    }

    // Optional: Helper to get specific LLVM types directly
    pub fn to_llvm_float_type<'ctx>(&self, context: &'ctx Context) -> Option<FloatType<'ctx>> {
        if *self == Type::Float {
            Some(context.f64_type())
        } else {
            None
        }
    }
    pub fn to_llvm_int_type<'ctx>(&self, context: &'ctx Context) -> Option<IntType<'ctx>> {
        if *self == Type::Int {
            Some(context.i64_type())
        } else {
            None
        }
    }
    // ... add more specific helpers as needed
}

// Implement Display for nice printing
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Float => write!(f, "float"),
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::Void => write!(f, "void"),
            Type::String => write!(f, "string"),
            Type::Vector(elem_ty) => write!(f, "vec<{}>", elem_ty), // Display vec<T>
        }
    }
}

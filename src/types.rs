// src/types.rs

use inkwell::context::Context;
use inkwell::types::{
    AnyTypeEnum, BasicTypeEnum, FloatType, IntType,
};
// Import LLVM types
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] // Use Eq and Hash for HashMap keys if needed
pub enum Type {
    Float, // Represents f64
    Int,   // Represents i64
    Bool,  // Represents i1 (boolean)
    Void,  // Represents lack of a value (for functions that don't return) - maybe add later
           // Add Function, Array, Struct types later
    String,
}

impl Type {
    /// Get the corresponding LLVM BasicTypeEnum
    /// Panics if called on Type::Void or other non-basic types later.
    pub fn to_llvm_basic_type<'ctx>(&self, context: &'ctx Context) -> BasicTypeEnum<'ctx> {
        match self {
            Type::Float => context.f64_type().into(),
            Type::Int => context.i64_type().into(), // Using 64-bit integers
            Type::Bool => context.bool_type().into(), // i1
            Type::Void => panic!("Cannot get LLVM BasicTypeEnum for Type::Void"),
            Type::String => panic!("Cannot get LLVM BasicTypeEnum for Type::String"),
        }
    }

    /// Get the corresponding LLVM AnyTypeEnum (includes void, functions etc.)
    pub fn to_llvm_any_type<'ctx>(&self, context: &'ctx Context) -> AnyTypeEnum<'ctx> {
        match self {
            Type::Float => context.f64_type().into(),
            Type::Int => context.i64_type().into(),
            Type::Bool => context.bool_type().into(),
            Type::Void => context.void_type().into(),
            Type::String => todo!()
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
        }
    }
}

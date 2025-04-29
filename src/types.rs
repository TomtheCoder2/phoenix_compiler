// src/types.rs

use inkwell::context::Context;
use inkwell::types::{AnyTypeEnum, BasicTypeEnum, FloatType, IntType};
use inkwell::AddressSpace;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq)] // Use Eq and Hash for HashMap keys if needed
pub enum Type {
    Float, // Represents f64
    Int,   // Represents i64
    Bool,  // Represents i1 (boolean)
    Void,  // Represents lack of a value (for functions that don't return) - maybe add later
    // Add Function, Array, Struct types later
    String,
    Vector(Box<Type>), // Added: Stores the element type, Boxed
    // Added: Represents a specific defined struct type
    Struct {
        name: Rc<String>, // Use Rc for cheap cloning of name
                          // Fields stored in definition, maybe just store name here?
                          // Let's store fields here for direct access during type checking.
                          // Use IndexMap to preserve field order for layout/codegen.
                          // todo:
                          //fields: Rc<IndexMap<String, Type>>, // Rc<IndexMap<FieldName, FieldType>>
    },
    // Maybe later: Generic Struct Instance? Type::StructInstance { base: Rc<String>, args: Vec<Type> }
    // Maybe later: Type::Function(...) signature type?
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Float => write!(f, "float"),
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::Void => write!(f, "void"),
            Type::String => write!(f, "string"),
            Type::Vector(elem_ty) => write!(f, "vec<{}>", elem_ty), // Display vec<T>
            // Type::Struct { name, fields } => {
            //     write!(f, "struct {} {{ ", name)?;
            //     for (i, (field_name, field_type)) in fields.iter().enumerate() {
            //         write!(f, "{}: {}", field_name, field_type)?;
            //         if i < fields.len() - 1 {
            //             write!(f, ", ")?;
            //         }
            //     }
            //     write!(f, " }}")
            // }
            Type::Struct { name } => write!(f, "{}", name),
        }
    }
}

impl Type {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    // pub order: usize, // Add order if using HashMap for fields
}

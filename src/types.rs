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
            Type::Struct { .. } => None, // Structs themselves aren't basic types
        }
    }

    /// Get the corresponding LLVM AnyTypeEnum (includes void, functions etc.)
    pub fn to_llvm_any_type<'ctx>(
        &self,
        context: &'ctx Context,
        struct_definitions: &HashMap<String, inkwell::types::StructType<'ctx>>,
    ) -> AnyTypeEnum<'ctx> {
        match self {
            Type::Float => context.f64_type().into(),
            Type::Int => context.i64_type().into(),
            Type::Bool => context.bool_type().into(),
            Type::Void => context.void_type().into(),
            Type::Struct { .. } => {
                // Return the LLVM StructType itself
                self.get_llvm_struct_type(context, struct_definitions)
                    .map(|st| st.into()) // Convert StructType to AnyTypeEnum
                    .unwrap_or_else(|| panic!("Struct type '{}' not found in LLVM definitions during AnyType conversion", self))
                // Should be found if TC passed
            }
            _ => panic!("Unhandled type in to_llvm_any_type: {:?}", self),
        }
    }

    // Get the LLVM StructType definition (requires lookup)
    // This needs access to the struct registry/definitions. Pass it in?
    pub fn get_llvm_struct_type<'ctx>(
        &self,
        context: &'ctx Context,
        struct_definitions: &HashMap<String, inkwell::types::StructType<'ctx>>,
    ) -> Option<inkwell::types::StructType<'ctx>> {
        match self {
            Type::Struct { name } => struct_definitions.get(&name.to_string()).copied(),
            _ => None,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    // pub order: usize, // Add order if using HashMap for fields
}

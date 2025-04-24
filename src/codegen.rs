// src/codegen.rs

use crate::ast::{
    type_node_to_type, BinaryOperator, ComparisonOperator, Expression, ExpressionKind,
    LogicalOperator, Program, Statement, StatementKind, TypeNode, TypeNodeKind, UnaryOperator,
};
use crate::location::Span;
// Added BasicValueEnum
use crate::types::Type;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::passes::{PassManager, PassManagerBuilder};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::types::{AnyType, BasicMetadataTypeEnum, BasicType};
use inkwell::values::{BasicMetadataValueEnum, BasicValue};
use inkwell::values::{BasicValueEnum, FunctionValue, GlobalValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use std::collections::HashMap;
use std::default::Default;
use std::ffi::CString;
use std::fmt;
use std::path::Path;

const RUNTIME_VEC_NEW: &str = "_phoenix_vec_new"; // fn(elem_size: i64, capacity: i64) -> i8*
const RUNTIME_VEC_GET_PTR: &str = "_phoenix_vec_get_ptr"; // fn(vec_handle: i8*, index: i64) -> i8* (pointer to element)
const RUNTIME_VEC_LEN: &str = "_phoenix_vec_len"; // fn(vec_handle: i8*) -> i64
const RUNTIME_VEC_PUSH: &str = "_phoenix_vec_push"; // fn(vec_handle: i8*, value_ptr: i8*) -> void
const RUNTIME_VEC_FREE: &str = "_phoenix_vec_free"; // fn(vec_handle: i8*) -> void
const RUNTIME_VEC_POP: &str = "_phoenix_vec_pop"; // fn(i8*) -> i8* (ptr to popped elem)

const RUNTIME_STR_NEW: &str = "_phoenix_str_new"; // fn(*const c_char) -> i8* handle
const RUNTIME_STR_LEN: &str = "_phoenix_str_len"; // fn(i8*) -> i64
const RUNTIME_STR_FREE: &str = "_phoenix_str_free"; // fn(i8*) -> void
const RUNTIME_STR_CONCAT: &str = "_phoenix_str_concat"; // fn(i8*, i8*) -> i8*
const RUNTIME_STR_GET_CHAR_PTR: &str = "_phoenix_str_get_char_ptr"; // fn(i8*, i64) -> i8*

// --- CodeGenError --- (UndefinedVariable is still relevant)
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenError {
    InvalidAstNode(String, Span),
    LlvmError(String, Span),
    // UnknownOperator(BinaryOperator), // Covered by expression compilation
    UndefinedVariable(String, Span),
    UndefinedFunction(String, Span),    // Added
    FunctionRedefinition(String, Span), // Added
    IncorrectArgumentCount {
        // Added
        func_name: String,
        expected: usize,
        found: usize,
        span: Span,
    },
    PrintArgError(String, Span),
    InvalidType(String, Span), // For type mismatches or unsupported operations
    InvalidUnaryOperation(String, Span), // For later
    InvalidBinaryOperation(String, Span), // For type mismatches
    MissingTypeInfo(String, Span), // If type info is needed but missing
    AssignmentToImmutable(String, Span),
    PrintStrArgError(String, Span),
    InvalidAssignmentTarget(String, Span),
}
impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenError::InvalidAstNode(msg, span) => {
                write!(f, "{}: Invalid AST Node: {}", span, msg)
            }
            CodeGenError::LlvmError(msg, span) => write!(f, "{}: LLVM Error: {}", span, msg),
            CodeGenError::UndefinedVariable(name, span) => {
                write!(f, "{}: Undefined variable: {}", span, name)
            }
            CodeGenError::UndefinedFunction(name, span) => {
                write!(f, "{}: Codegen Error: Undefined function '{}'", span, name)
            }
            CodeGenError::FunctionRedefinition(name, span) => {
                write!(f, "{}: Codegen Error: Function '{}' redefined", span, name)
            }
            CodeGenError::IncorrectArgumentCount {
                func_name,
                expected,
                found,
                span,
            } => write!(
                f,
                "{}: Codegen Error: Call to function '{}' expected {} arguments, found {}",
                span, func_name, expected, found
            ),
            CodeGenError::PrintArgError(msg, span) => {
                write!(f, "{}: Built-in 'print' Error: {}", span, msg)
            }
            CodeGenError::InvalidType(msg, span) => {
                write!(f, "{}: Codegen Type Error: {}", span, msg)
            }
            CodeGenError::InvalidBinaryOperation(msg, span) => {
                write!(f, "{}: Codegen Binary Op Error: {}", span, msg)
            }
            CodeGenError::MissingTypeInfo(name, span) => {
                write!(
                    f,
                    "{}: Codegen Error: Missing type info for '{}'",
                    span, name
                )
            }
            CodeGenError::InvalidUnaryOperation(msg, span) => {
                write!(f, "{}: Codegen Unary Op Error: {}", span, msg)
            }
            CodeGenError::AssignmentToImmutable(msg, span) => {
                write!(
                    f,
                    "{}: Codegen Error: Cannot assign to immutable variable: {}",
                    span, msg
                )
            }
            CodeGenError::PrintStrArgError(msg, span) => {
                write!(
                    f,
                    "{}: Codegen Error: print_str argument error: {}",
                    span, msg
                )
            }
            CodeGenError::InvalidAssignmentTarget(msg, span) => {
                write!(
                    f,
                    "{}: Codegen Error: Invalid assignment target: {}",
                    span, msg
                )
            }
        }
    }
}
// Result now often returns BasicValueEnum as expressions can yield int, float, or bool
type CompileExprResult<'ctx> = Result<BasicValueEnum<'ctx>, CodeGenError>;
// CompileStmtResult remains similar, might yield BasicValueEnum if expr stmt returns non-float
type CompileStmtResult<'ctx> = Result<(Option<BasicValueEnum<'ctx>>, bool), CodeGenError>;

// Helper struct to store variable info (pointer + type)
#[derive(Debug, Clone)] // Copy is efficient as PointerValue/Type are Copy
                        // todo add def_span
struct VariableInfo<'ctx> {
    ptr: PointerValue<'ctx>,
    ty: Type,
    is_mutable: bool,
}

pub struct Compiler<'a, 'ctx> {
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    // Symbol table now stores VariableInfo
    variables: HashMap<String, VariableInfo<'ctx>>,
    // Function map stores signature (params + return) along with FunctionValue
    // Using our Type enum for the signature representation.
    pub(crate) functions: HashMap<String, (Vec<Type>, Type, FunctionValue<'ctx>)>,
    current_function_return_type: Option<Type>, // Track expected return type
    fpm: PassManager<FunctionValue<'ctx>>,
    // Keep print wrapper stuff
    current_function: Option<FunctionValue<'ctx>>,
    print_float_wrapper_func: Option<FunctionValue<'ctx>>,
    print_int_wrapper_func: Option<FunctionValue<'ctx>>,
    print_bool_wrapper_func: Option<FunctionValue<'ctx>>,
    print_str_wrapper_func: Option<FunctionValue<'ctx>>, // Added cache
    print_str_ln_wrapper_func: Option<FunctionValue<'ctx>>,
    // vec cache
    vec_new_func: Option<FunctionValue<'ctx>>,
    vec_get_ptr_func: Option<FunctionValue<'ctx>>,
    vec_len_func: Option<FunctionValue<'ctx>>,
    vec_push_func: Option<FunctionValue<'ctx>>,
    vec_free_func: Option<FunctionValue<'ctx>>,
    vec_pop_func: Option<FunctionValue<'ctx>>,
    // string cache
    str_new_func: Option<FunctionValue<'ctx>>,
    str_len_func: Option<FunctionValue<'ctx>>,
    str_free_func: Option<FunctionValue<'ctx>>,
    str_concat_func: Option<FunctionValue<'ctx>>,
    str_get_char_ptr_func: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    pub fn new(
        context: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
    ) -> Self {
        // --- Initialize the Pass Manager ---
        let pass_manager_builder = PassManagerBuilder::create();
        // Configure standard optimization passes based on a level.
        // Level 2 (`Default`) is a good balance. Level 3 (`Aggressive`) is more intense.
        pass_manager_builder.set_optimization_level(OptimizationLevel::Default);

        // Create the Function Pass Manager
        let fpm = PassManager::create(module); // PassManager::create requires module for Module passes

        // Add desired passes - PassManagerBuilder helps populate common ones
        // You can also add passes manually using fpm.add_..._pass() methods.
        // See inkwell docs / LLVM pass list for available passes.
        // Standard sequence for O2/O3 often includes:
        // - mem2reg: Promotes memory variables (allocas) to SSA registers - HUGE gains
        // - instcombine: Combines redundant instructions
        // - gvn: Global Value Numbering (eliminates redundant calculations)
        // - simplifycfg: Simplifies control flow graph
        // - ... and many more

        // Populate the FPM using the builder's recommendations for the chosen opt level
        pass_manager_builder.populate_function_pass_manager(&fpm);

        // It's crucial to initialize the FPM *after* adding passes.
        fpm.initialize(); // Call initialize AFTER passes are added

        Compiler {
            context,
            builder,
            module,
            variables: HashMap::new(),
            functions: HashMap::new(),
            fpm,                                // Store the initialized FPM
            current_function_return_type: None, // Initialize
            current_function: None,
            print_float_wrapper_func: None,
            print_int_wrapper_func: None,
            print_bool_wrapper_func: None,
            print_str_wrapper_func: None, // Added cache
            print_str_ln_wrapper_func: None,
            vec_new_func: None,
            vec_get_ptr_func: None,
            vec_len_func: None,
            vec_push_func: None,
            vec_free_func: None, // Init new caches
            vec_pop_func: None,
            str_new_func: None,
            str_len_func: None,
            str_free_func: None,
            str_concat_func: None,
            str_get_char_ptr_func: None,
        }
    }

    /// Compile the program into the module, generating a standard C main function
    /// that calls printf with the result of the last top-level expression.
    pub fn compile_program_to_module(&mut self, program: &Program) -> Result<(), CodeGenError> {
        // --- Reset state for this compilation unit ---
        self.functions.clear(); // Clear user function definitions (printf is handled separately)
        self.print_float_wrapper_func = None;
        self.print_int_wrapper_func = None;
        self.print_bool_wrapper_func = None;
        self.print_str_wrapper_func = None; // Reset string wrapper cache too
        self.print_str_ln_wrapper_func = None;
        self.vec_new_func = None;
        self.vec_get_ptr_func = None;
        self.vec_len_func = None;
        self.vec_push_func = None;
        self.vec_free_func = None;
        self.vec_pop_func = None;
        self.str_new_func = None;
        self.str_len_func = None;
        self.str_free_func = None;
        self.str_concat_func = None;
        self.str_get_char_ptr_func = None;
        self.current_function = None; // Reset current function

        // --- Declare Printf Early ---
        // Ensure printf is declared before compiling user code that might need it (though not used directly by user code yet)

        // --- Define User Functions ---
        // First pass: Define all user functions so they are available for calls
        // Need to be careful about compile_statement potentially modifying state needed later
        // Let's compile functions within the main loop for now, requires functions defined before use.
        // A multi-pass approach might be better later.

        // --- Setup C-Style Main Function ---
        let i32_type = self.context.i32_type();
        // Standard C main signature: int main() (or int main(int argc, char** argv))
        // We'll use the simple int main() for now.
        let main_fn_type = i32_type.fn_type(&[], false);
        let main_function = self.module.add_function(
            "main", // Standard C entry point name
            main_fn_type,
            None, // Default linkage (usually external)
        );
        let entry_block = self.context.append_basic_block(main_function, "entry");

        // --- Save/Restore context ---
        let original_builder_pos = self.builder.get_insert_block();
        let original_func = self.current_function;
        let original_vars = self.variables.clone();
        self.builder.position_at_end(entry_block);
        self.current_function = Some(main_function);
        self.variables.clear();

        // --- Compile Top-Level Statements ---
        // let mut last_main_expr_value: Option<FloatValue<'ctx>> = None;
        for stmt in &program.statements {
            self.compile_statement(stmt)?;
            // match self.compile_statement(stmt)? {
            //     Some(value) => last_main_expr_value = Some(value),
            //     None => {} // LetBinding or FunctionDef
            // }
        }

        // --- Build Return for C Main (return 0) ---
        self.builder
            .build_return(Some(&i32_type.const_int(0, false))); // return 0;

        // --- Restore Compiler State ---
        // ... (restore variables, current_function, builder position) ...
        self.variables = original_vars;
        self.current_function = original_func;
        if let Some(bb) = original_builder_pos {
            self.builder.position_at_end(bb);
        }

        // --- Optimize and Verify Main ---
        // No need to optimize here if we optimize later or let TargetMachine handle it?
        // Let's keep the FPM run on main for consistency for now.
        if main_function.verify(true) {
            let changed = self.fpm.run_on(&main_function);
            if changed {
                println!("Optimizer changed function 'main'");
            }
            if !main_function.verify(true) {
                // Handle verification failure after optimization
                // unsafe { main_function.delete(); } // No need to delete, just return Err
                return Err(CodeGenError::LlvmError(
                    "Main function verification failed after optimization".to_string(),
                    Span::default(), // Placeholder, should be the actual span of the main function
                ));
            }
        } else {
            // Handle verification failure before optimization
            // unsafe { main_function.delete(); }
            // write the IR to file for debugging
            let ir_file_path = Path::new("main_verification_error.ll");
            match std::fs::write(&ir_file_path, self.module.print_to_string().to_string()) {
                Ok(_) => println!("LLVM IR written to: {}", ir_file_path.display()),
                Err(e) => eprintln!("Error writing LLVM IR to file: {}", e),
            }

            return Err(CodeGenError::LlvmError(
                "Main function verification failed before optimization".to_string(),
                Span::default(), // Placeholder, should be the actual span of the main function
            ));
        }

        Ok(()) // Module is ready
    }

    // Helper to get the type of an expression - NEEDS TYPE CHECKING PHASE ideally
    // For now, we infer based on structure, which is LIMITED and UNSAFE.
    fn infer_expression_type(&self, expr: &Expression) -> Result<Type, CodeGenError> {
        let span = expr.span.clone();
        match &expr.kind {
            ExpressionKind::FloatLiteral(_) => Ok(Type::Float),
            ExpressionKind::IntLiteral(_) => Ok(Type::Int),
            ExpressionKind::BoolLiteral(_) => Ok(Type::Bool),
            ExpressionKind::Variable(name) => {
                self.variables
                    .get(name)
                    .map(|info| info.ty.clone()) // Get type from symbol table
                    .ok_or_else(|| CodeGenError::MissingTypeInfo(name.clone(), span))
                // Error if var used before defined (or type missing)
            }
            ExpressionKind::BinaryOp { op, left, right } => {
                // Very basic inference: assume result type matches operands
                // THIS IS WRONG: needs proper type checking based on operator rules!
                // E.g., int + int -> int, float + float -> float
                self.infer_expression_type(left)
            }
            ExpressionKind::ComparisonOp { .. } => {
                // Comparisons always return boolean
                Ok(Type::Bool)
            }
            ExpressionKind::FunctionCall { name, .. } => {
                // Look up function's declared return type
                self.functions
                    .get(name)
                    .map(|(_, ret_ty, _)| ret_ty.clone()) // Get return type from map
                    .ok_or_else(|| {
                        CodeGenError::MissingTypeInfo(format!("function {}", name), span)
                    })
            }
            _ => Err(CodeGenError::LlvmError(
                "unexpected branch".to_string(),
                span,
            )),
        }
    }

    // Helper to create a global string constant (for format strings)
    fn create_global_string(&self, name: &str, value: &str, linkage: Linkage) -> GlobalValue<'ctx> {
        let c_string = CString::new(value).expect("CString::new failed");
        // Use `to_bytes()` (length *without* null) and let LLVM add the null terminator (`false`)
        // This ensures the type has the correct size (e.g., [4 x i8] for "%f\n")
        let string_val = self
            .context
            .const_string(c_string.to_bytes_with_nul(), false);

        let global = self.module.add_global(
            string_val.get_type(), // Type from const_string should be correct now
            Some(AddressSpace::default()),
            name,
        );
        global.set_linkage(linkage);
        global.set_initializer(&string_val);
        global.set_constant(true);
        global
    }

    // Helper to create an alloca instruction in the function's entry block
    // This ensures all allocas happen at the start, which is good practice in LLVM.
    // Alloca helper now needs the phoenix Type to allocate correctly
    fn create_entry_block_alloca(&self, name: &str, ty: Type) -> PointerValue<'ctx> {
        let temp_builder = self.context.create_builder();
        // Ensure current_function is set before calling this
        let entry_block = self
            .current_function
            .expect("No current function for alloca")
            .get_first_basic_block()
            .unwrap();
        match entry_block.get_first_instruction() {
            Some(first_instr) => temp_builder.position_before(&first_instr),
            None => temp_builder.position_at_end(entry_block),
        };
        // Allocate memory for the correct LLVM type based on phoenix Type
        let llvm_type = match ty.to_llvm_basic_type(self.context) {
            Some(llvm_type) => llvm_type,
            None => {
                panic!("Invalid type for variable '{}'", name);
            }
        };
        // todo: check if unwrap is safe
        temp_builder.build_alloca(llvm_type, name).unwrap()
    }

    // Helper to compile an L-Value expression into a *pointer* to the location.
    // Returns PointerValue pointing to the final element/variable.
    fn compile_lvalue_pointer(
        &mut self,
        target_expr: &Expression,
    ) -> Result<PointerValue<'ctx>, CodeGenError> {
        let span = target_expr.span.clone();
        match &target_expr.kind {
            ExpressionKind::Variable(name) => {
                // Return the pointer stored in the symbol table (from alloca)
                self.variables
                    .get(name)
                    .map(|info| info.ptr)
                    .ok_or_else(|| {
                        CodeGenError::UndefinedVariable(name.clone(), target_expr.span.clone())
                    }) // Should be caught by TC
            }
            ExpressionKind::IndexAccess {
                target: vec_expr,
                index,
            } => {
                // 1. Compile the inner target expression to get the vector handle (i8*)
                let vec_handle_val = self.compile_expression(vec_expr)?;
                let vec_handle_ptr = match vec_handle_val {
                    BasicValueEnum::PointerValue(pv) => pv,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            "Expected vector handle (pointer) for index target".into(),
                            vec_expr.span.clone(),
                        ))
                    }
                };

                // 2. Compile the index expression (expect i64)
                let index_val = self.compile_expression(index)?;
                let index_i64 = match index_val {
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 64 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            "Index must be an integer".into(),
                            index.span.clone(),
                        ))
                    }
                };

                // 3. Get element type from vec_expr's resolved type
                let target_resolved_type = vec_expr.get_type();
                let elem_toy_type = match target_resolved_type {
                    Some(Type::Vector(et)) => *et,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            "Expected vector type for index access".into(),
                            vec_expr.span.clone(),
                        ))
                    }
                };
                let elem_llvm_type = elem_toy_type.to_llvm_basic_type(self.context).unwrap();

                // 4. Call runtime vec_get_ptr(handle, index) -> i8*
                let vec_get_ptr_func = self.get_or_declare_vec_get_ptr();
                let elem_ptr_i8 = match self.builder.build_call(
                    vec_get_ptr_func,
                    &[vec_handle_ptr.into(), index_i64.into()],
                    "elem_ptr_i8",
                ) {
                    Ok(call) => call
                        .try_as_basic_value()
                        .left()
                        .ok_or(CodeGenError::LlvmError(
                            "Failed to get element pointer from vec_get_ptr".to_string(),
                            span.clone(),
                        ))?
                        .into_pointer_value(), // i8* pointer to the element
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("Call to vec_get_ptr failed: {}", e),
                            span.clone(),
                        ))
                    }
                };
                // TODO: Check for null ptr return from vec_get_ptr (bounds error)

                // 5. Cast i8* element slot pointer to actual element type pointer (e.g., i64*, float*)
                let elem_ptr_typed = match self.builder.build_pointer_cast(
                    elem_ptr_i8,
                    elem_llvm_type.ptr_type(AddressSpace::default()),
                    "elem_ptr_typed",
                ) {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("Pointer cast failed: {}", e),
                            span,
                        ))
                    }
                };

                // 6. Return the typed pointer to the element
                Ok(elem_ptr_typed)
            }
            _ => Err(CodeGenError::InvalidAssignmentTarget(
                "Target expression is not assignable".into(),
                span,
            )),
        }
    }

    /// Compiles a single Expression node (used by statements)
    /// Returns a FloatValue representing the result of the expression.
    pub fn compile_expression(&mut self, expr: &Expression) -> CompileExprResult<'ctx> {
        let span = expr.span.clone();
        match &expr.kind {
            ExpressionKind::FloatLiteral(value) => {
                Ok(self.context.f64_type().const_float(*value).into())
            }
            ExpressionKind::IntLiteral(value) => {
                // Assuming i64 for now
                Ok(self
                    .context
                    .i64_type()
                    .const_int(*value as u64, true)
                    .into()) // true=signed
            }
            ExpressionKind::BoolLiteral(value) => {
                Ok(self
                    .context
                    .bool_type()
                    .const_int(*value as u64, false)
                    .into()) // false=unsigned
            }

            ExpressionKind::Variable(name) => {
                match self.variables.get(name) {
                    Some(info) => {
                        let llvm_type = match info.ty.to_llvm_basic_type(self.context) {
                            Some(llvm_type) => llvm_type,
                            None => {
                                return Err(CodeGenError::InvalidType(
                                    format!("Invalid type for variable '{}'", name),
                                    span,
                                ))
                            }
                        };
                        let loaded_val = self
                            .builder
                            .build_load(llvm_type, info.ptr, name)
                            .map_err(|e| {
                                CodeGenError::LlvmError(format!("Load failed: {}", e), span)
                            })?;
                        Ok(loaded_val) // Returns BasicValueEnum
                    }
                    None => Err(CodeGenError::UndefinedVariable(name.clone(), span)),
                }
            }

            ExpressionKind::StringLiteral(value) => {
                // 1. Create global constant for the *initial value* only
                let global_name = format!("g_init_str_{}", self.module.get_globals().count());
                let global_val = self.create_global_string(&global_name, value, Linkage::Private);
                let zero_idx = self.context.i32_type().const_int(0, false);
                let string_type = global_val.get_value_type().into_array_type();
                // 2. Get i8* pointer to the global constant data
                let c_str_ptr = match unsafe {
                    self.builder.build_in_bounds_gep(
                        string_type,
                        global_val.as_pointer_value(),
                        &[zero_idx, zero_idx],
                        "init_str_ptr",
                    )
                } {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(format!("GEP failed: {}", e), span))
                    }
                };
                // 3. Call runtime _toylang_str_new(c_str_ptr)
                let str_new_func = self.get_or_declare_str_new();
                let str_handle = match self.builder.build_call(
                    str_new_func,
                    &[c_str_ptr.into()],
                    "new_str_handle",
                ) {
                    Ok(call) => call
                        .try_as_basic_value()
                        .left()
                        .ok_or(CodeGenError::LlvmError(
                            "Failed to get string handle from call".to_string(),
                            span,
                        ))?
                        .into_pointer_value(), // i8* handle
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("Call to _phoenix_str_new failed: {}", e),
                            span,
                        ))
                    }
                };
                // 4. Expression evaluates to the string handle (i8*)
                Ok(str_handle.into())
            }

            // Arithmetic Operations (Example: requires operands to be the same type for now)
            ExpressionKind::BinaryOp { op, left, right } => {
                let lhs = self.compile_expression(left)?;
                let rhs = self.compile_expression(right)?;

                let lhs_type = left.get_type().unwrap_or(Type::Void); // Get type from AST node
                let rhs_type = right.get_type().unwrap_or(Type::Void);

                // Basic type checking (Replace with proper type checker later)
                match (lhs_type, rhs_type, op) {
                    (Type::Int, Type::Int, _) => {
                        let l = lhs.into_int_value();
                        let r = rhs.into_int_value();
                        let result = match match op {
                            BinaryOperator::Add => self.builder.build_int_add(l, r, "addtmp"),
                            BinaryOperator::Subtract => self.builder.build_int_sub(l, r, "subtmp"),
                            BinaryOperator::Multiply => self.builder.build_int_mul(l, r, "multmp"),
                            BinaryOperator::Divide => {
                                self.builder.build_int_signed_div(l, r, "sdivtmp")
                            } // Signed division
                        } {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during integer operation: {}", e),
                                    span,
                                ))
                            }
                        };
                        Ok(result.into())
                    }
                    (Type::Float, Type::Float, _) => {
                        let l = lhs.into_float_value();
                        let r = rhs.into_float_value();
                        let result = match match op {
                            BinaryOperator::Add => self.builder.build_float_add(l, r, "faddtmp"),
                            BinaryOperator::Subtract => {
                                self.builder.build_float_sub(l, r, "fsubtmp")
                            }
                            BinaryOperator::Multiply => {
                                self.builder.build_float_mul(l, r, "fmultmp")
                            }
                            BinaryOperator::Divide => self.builder.build_float_div(l, r, "fdivtmp"),
                        } {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during float operation: {}", e),
                                    span,
                                ))
                            }
                        };
                        Ok(result.into())
                    }
                    (Type::String, Type::String, BinaryOperator::Add) => {
                        let str_concat_func = self.get_or_declare_str_concat();
                        let result_handle = match self.builder.build_call(
                            str_concat_func,
                            &[lhs.into(), rhs.into()],
                            "concat_str",
                        ) {
                            Ok(call) => call
                                .try_as_basic_value()
                                .left()
                                .ok_or(CodeGenError::LlvmError(
                                    "Failed to get string handle from call".to_string(),
                                    span,
                                ))?
                                .into_pointer_value(), // i8* handle
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("Call to _phoenix_str_concat failed: {}", e),
                                    span,
                                ))
                            }
                        };
                        Ok(result_handle.into()) // Result is new string handle (i8*)
                    }
                    _ => Err(CodeGenError::InvalidBinaryOperation(
                        format!(
                            "Type mismatch or unsupported types for operator {} (lhs: {}, rhs: {})",
                            op,
                            lhs.get_type(),
                            rhs.get_type()
                        ),
                        span,
                    )),
                }
            }

            // Comparison Operations (Result is always Bool i1)
            ExpressionKind::ComparisonOp { op, left, right } => {
                let lhs = self.compile_expression(left)?;
                let rhs = self.compile_expression(right)?;

                // Determine comparison predicate based on operand types
                let predicate = match match (lhs, rhs) {
                    // Integer comparison
                    (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                        let llvm_pred = match op {
                            ComparisonOperator::Equal => IntPredicate::EQ,
                            ComparisonOperator::NotEqual => IntPredicate::NE,
                            ComparisonOperator::LessThan => IntPredicate::SLT, // Signed Less Than
                            ComparisonOperator::LessEqual => IntPredicate::SLE, // Signed Less Or Equal
                            ComparisonOperator::GreaterThan => IntPredicate::SGT, // Signed Greater Than
                            ComparisonOperator::GreaterEqual => IntPredicate::SGE, // Signed Greater Or Equal
                        };
                        self.builder.build_int_compare(llvm_pred, l, r, "icmptmp")
                    }
                    // Float comparison
                    (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                        let llvm_pred = match op {
                            ComparisonOperator::Equal => FloatPredicate::OEQ, // Ordered Equal
                            ComparisonOperator::NotEqual => FloatPredicate::ONE, // Ordered Not Equal
                            ComparisonOperator::LessThan => FloatPredicate::OLT, // Ordered Less Than
                            ComparisonOperator::LessEqual => FloatPredicate::OLE, // Ordered Less Or Equal
                            ComparisonOperator::GreaterThan => FloatPredicate::OGT, // Ordered Greater Than
                            ComparisonOperator::GreaterEqual => FloatPredicate::OGE, // Ordered Greater Or Equal
                        };
                        self.builder.build_float_compare(llvm_pred, l, r, "fcmptmp")
                    }
                    // Boolean comparison (only ==, != makes sense)
                    (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r))
                        if l.get_type().get_bit_width() == 1 =>
                    {
                        match op {
                            ComparisonOperator::Equal => self.builder.build_int_compare(
                                IntPredicate::EQ,
                                l,
                                r,
                                "icmp_bool_eq",
                            ),
                            ComparisonOperator::NotEqual => self.builder.build_int_compare(
                                IntPredicate::NE,
                                l,
                                r,
                                "icmp_bool_ne",
                            ),
                            _ => {
                                return Err(CodeGenError::InvalidBinaryOperation(
                                    format!("Unsupported comparison {} for booleans", op),
                                    span,
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(CodeGenError::InvalidBinaryOperation(
                            format!(
                                "Type mismatch for comparison operator {} (lhs: {}, rhs: {})",
                                op,
                                lhs.get_type(),
                                rhs.get_type()
                            ),
                            span,
                        ))
                    }
                } {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("LLVM error during comparison operation: {}", e),
                            span,
                        ))
                    }
                };
                Ok(predicate.into()) // Comparison result is IntValue (i1)
            }
            // --- Unary Operation ---
            ExpressionKind::UnaryOp { op, operand } => {
                let compiled_operand = self.compile_expression(operand)?;

                match op {
                    UnaryOperator::Negate => {
                        match compiled_operand {
                            BasicValueEnum::IntValue(iv) => {
                                // LLVM has 'neg', but sometimes it's emitted as 'sub 0, op'
                                // build_int_neg should handle this correctly.
                                Ok((match self.builder.build_int_neg(iv, "ineg_tmp") {
                                    Ok(val) => val,
                                    Err(e) => {
                                        return Err(CodeGenError::LlvmError(
                                            format!("LLVM error during integer negation: {}", e),
                                            span,
                                        ))
                                    }
                                })
                                .into())
                            }
                            BasicValueEnum::FloatValue(fv) => {
                                Ok((match self.builder.build_float_neg(fv, "fneg_tmp") {
                                    Ok(val) => val,
                                    Err(e) => {
                                        return Err(CodeGenError::LlvmError(
                                            format!("LLVM error during float negation: {}", e),
                                            span,
                                        ))
                                    }
                                })
                                .into())
                            }
                            _ => Err(CodeGenError::InvalidUnaryOperation(
                                format!(
                                    "Cannot apply arithmetic negate '-' to type {}",
                                    compiled_operand.get_type()
                                ),
                                span,
                            )),
                        }
                    }
                    UnaryOperator::Not => {
                        match compiled_operand {
                            // Expecting i1 for logical not
                            BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => {
                                // LLVM doesn't have a native boolean 'not'. Common ways:
                                // 1. xor with true (1): not x == x ^ 1
                                // 2. Compare with false (0): not x == (x == 0)
                                // Let's use XOR.
                                let true_val = self.context.bool_type().const_int(1, false);
                                Ok((match self.builder.build_xor(iv, true_val, "not_tmp") {
                                    Ok(val) => val,
                                    Err(e) => {
                                        return Err(CodeGenError::LlvmError(
                                            format!("LLVM error during boolean negation: {}", e),
                                            span,
                                        ))
                                    }
                                })
                                .into())
                            }
                            _ => Err(CodeGenError::InvalidUnaryOperation(
                                format!(
                                    "Cannot apply logical not '!' to type {}",
                                    compiled_operand.get_type()
                                ),
                                span,
                            )),
                        }
                    }
                } // End match op
            }
            // --- Logical Operators (with short-circuiting) ---
            ExpressionKind::LogicalOp { op, left, right } => {
                let current_func = self
                    .current_function
                    .expect("Cannot compile logical op outside func");

                match op {
                    LogicalOperator::And => {
                        // --- Short-circuiting AND (a && b) ---
                        // 1. Evaluate LHS (a)
                        let lhs_val = self.compile_expression(left)?;
                        let lhs_bool = match lhs_val {
                            // Check if lhs_val is i1 (boolean)
                            BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => {
                                iv
                            }
                            _ => {
                                return Err(CodeGenError::InvalidBinaryOperation(
                                    format!(
                                        "Left operand of '&&' must be boolean (i1), found {}",
                                        lhs_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        };
                        let cond_bb = self
                            .builder
                            .get_insert_block()
                            .expect("Builder not positioned before AND op");

                        // 2. Create blocks
                        let rhs_bb = self.context.append_basic_block(current_func, "and_rhs");
                        let merge_bb = self.context.append_basic_block(current_func, "and_cont");

                        // 3. Conditional branch based on LHS
                        // If LHS is false, jump directly to merge (result is false)
                        // If LHS is true, jump to RHS block to evaluate RHS
                        let false_val = self.context.bool_type().const_int(0, false);
                        self.builder
                            .build_conditional_branch(lhs_bool, rhs_bb, merge_bb);

                        // 4. Compile RHS block
                        self.builder.position_at_end(rhs_bb);
                        let rhs_val = self.compile_expression(right)?;
                        let rhs_bool = match rhs_val {
                            // Check if rhs_val is i1 (boolean)
                            BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => {
                                iv
                            }
                            _ => {
                                return Err(CodeGenError::InvalidBinaryOperation(
                                    format!(
                                        "Right operand of '&&' must be boolean (i1), found {}",
                                        rhs_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        };
                        let rhs_final_bb = self.builder.get_insert_block().unwrap_or(rhs_bb); // Block where RHS value is available
                        if rhs_final_bb.get_terminator().is_none() {
                            self.builder.build_unconditional_branch(merge_bb); // Terminate RHS path
                        }
                        // Branch to merge block (result is RHS value)
                        if rhs_bb.get_terminator().is_none() {
                            self.builder.build_unconditional_branch(merge_bb);
                        }
                        // let rhs_end_bb = self.builder.get_insert_block().unwrap_or(rhs_bb);

                        // 5. Compile Merge Block (PHI node)
                        self.builder.position_at_end(merge_bb);
                        let phi = match self.builder.build_phi(self.context.bool_type(), "and_tmp")
                        {
                            Ok(phi) => phi,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during PHI node creation: {}", e),
                                    span,
                                ))
                            }
                        };
                        phi.add_incoming(&[
                            (&false_val, cond_bb),     // If branch from cond was false, result is false
                            (&rhs_bool, rhs_final_bb), // If branch from rhs_bb, result is rhs_bool
                        ]);

                        Ok(phi.as_basic_value())
                    } // End And

                    LogicalOperator::Or => {
                        // --- Short-circuiting OR (a || b) ---
                        // 1. Evaluate LHS (a)
                        let lhs_val = self.compile_expression(left)?;
                        let lhs_bool = match lhs_val {
                            // Check if lhs_val is i1 (boolean)
                            BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => {
                                iv
                            }
                            _ => {
                                return Err(CodeGenError::InvalidBinaryOperation(
                                    format!(
                                        "Left operand of '||' must be boolean (i1), found {}",
                                        lhs_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        };

                        // 2. Create blocks
                        let rhs_bb = self.context.append_basic_block(current_func, "or_rhs");
                        let merge_bb = self.context.append_basic_block(current_func, "or_cont");

                        // 3. Conditional branch based on LHS
                        // If LHS is true, jump directly to merge (result is true)
                        // If LHS is false, jump to RHS block to evaluate RHS
                        let true_val = self.context.bool_type().const_int(1, false);
                        // Branch needs block where LHS was computed for PHI later
                        let cond_bb = self
                            .builder
                            .get_insert_block()
                            .expect("Builder not positioned before OR op");
                        self.builder
                            .build_conditional_branch(lhs_bool, merge_bb, rhs_bb); // Note order: true->merge, false->rhs

                        // 4. Compile RHS block
                        self.builder.position_at_end(rhs_bb);
                        let rhs_val = self.compile_expression(right)?;
                        let rhs_bool = match rhs_val {
                            // Check if rhs_val is i1 (boolean)
                            BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => {
                                iv
                            }
                            _ => {
                                return Err(CodeGenError::InvalidBinaryOperation(
                                    format!(
                                        "Right operand of '||' must be boolean (i1), found {}",
                                        rhs_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        };
                        let rhs_final_bb = self.builder.get_insert_block().unwrap_or(rhs_bb); // Block where RHS value is available
                        if rhs_final_bb.get_terminator().is_none() {
                            self.builder.build_unconditional_branch(merge_bb); // Terminate RHS path
                        }
                        // Branch to merge block (result is RHS value)
                        if rhs_bb.get_terminator().is_none() {
                            self.builder.build_unconditional_branch(merge_bb);
                        }
                        let rhs_end_bb = self.builder.get_insert_block().unwrap_or(rhs_bb);

                        // 5. Compile Merge Block (PHI node)
                        self.builder.position_at_end(merge_bb);
                        let phi = match self.builder.build_phi(self.context.bool_type(), "or_tmp") {
                            Ok(phi) => phi,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during PHI node creation: {}", e),
                                    span,
                                ))
                            }
                        };
                        phi.add_incoming(&[
                            (&true_val, cond_bb),      // If branch from cond was true, result is true
                            (&rhs_bool, rhs_final_bb), // If branch from rhs_bb, result is rhs_bool
                        ]);

                        Ok(phi.as_basic_value())
                    } // End Or
                } // End match op
            } // End LogicalOp
            // --- Handle Function Calls ---
            ExpressionKind::FunctionCall { name, args } => {
                // --- >> SPECIAL CASE: Built-in print function << ---
                if name == "print"
                    || name == "print_int"
                    || name == "print_str"
                    || name == "print_bool"
                    || name == "print_float"
                    || name == "println"
                {
                    if args.len() != 1 {
                        return Err(CodeGenError::IncorrectArgumentCount {
                            func_name: name.to_string(),
                            expected: 1,
                            found: args.len(),
                            span,
                        });
                    }

                    let arg_value = self.compile_expression(&args[0])?;

                    // Determine if we need a newline
                    let add_newline = name == "println";

                    // For print_int and print_str, validate the argument type
                    if name == "print_int" {
                        match arg_value {
                            BasicValueEnum::IntValue(_) => {}
                            _ => {
                                return Err(CodeGenError::PrintArgError(
                                    format!(
                                        "print_int expects an integer argument, found {}",
                                        arg_value.get_type()
                                    ),
                                    span,
                                ))
                            }
                        }
                    } else if name == "print_str" {
                        match arg_value {
                            BasicValueEnum::PointerValue(pv)
                            if pv.get_type()
                                == self.context.i8_type().ptr_type(AddressSpace::default()) => {}
                            _ => {
                                return Err(CodeGenError::PrintStrArgError(
                                    format!(
                                        "Argument must be a string literal (evaluating to i8*), found {}",
                                        arg_value.get_type()
                                    ),
                                    span,
                                ))
                            }
                        }
                    }

                    // Choose appropriate wrapper based on argument type
                    match arg_value {
                        BasicValueEnum::IntValue(iv) => {
                            if iv.get_type().get_bit_width() == 1 {
                                // Boolean
                                let wrapper = self.get_or_declare_print_bool_wrapper()?;
                                self.builder.build_call(wrapper, &[iv.into()], "");
                            } else {
                                // Integer (i64)
                                let wrapper = self.get_or_declare_print_int_wrapper()?;
                                self.builder.build_call(wrapper, &[iv.into()], "");
                            }
                        }
                        BasicValueEnum::FloatValue(fv) => {
                            let wrapper = self.get_or_declare_print_float_wrapper()?;
                            self.builder.build_call(wrapper, &[fv.into()], "");
                        }
                        BasicValueEnum::PointerValue(pv) => {
                            // Check if it's a string pointer (i8*)
                            if pv.get_type()
                                == self.context.i8_type().ptr_type(AddressSpace::default())
                            {
                                let wrapper = self.get_or_declare_print_str_wrapper()?;
                                self.builder.build_call(wrapper, &[pv.into()], "");
                            } else {
                                return Err(CodeGenError::PrintArgError(
                                    "Cannot print raw pointers directly".to_string(),
                                    span,
                                ));
                            }
                        }
                        _ => {
                            return Err(CodeGenError::PrintArgError(
                                "Unsupported type for print".to_string(),
                                span,
                            ));
                        }
                    }

                    // If println, add a newline
                    if add_newline {
                        // Get or declare the print_str_ln wrapper instead of creating a new global string each time
                        let print_str_ln_wrapper = self.get_or_declare_print_str_ln_wrapper()?;
                        self.builder.build_call(print_str_ln_wrapper, &[], "");
                    }

                    // Print functions return 0.0 float for now
                    Ok(self.context.f64_type().const_float(0.0).into())
                } else if name == "push" {
                    // Args validated by type checker: args[0] is vec handle, args[1] is element
                    let vec_handle_val = self.compile_expression(&args[0])?;
                    let elem_val = self.compile_expression(&args[1])?;
                    let vec_handle_ptr = match vec_handle_val {
                        BasicValueEnum::PointerValue(pv)
                            if pv.get_type()
                                == self.context.i8_type().ptr_type(AddressSpace::default()) =>
                        {
                            pv
                        }
                        _ => {
                            return Err(CodeGenError::InvalidType(
                                format!(
                                    "Expected vector handle (i8*), found {}",
                                    vec_handle_val.get_type()
                                ),
                                span,
                            ))
                        }
                    };

                    // Allocate temp space for element, store it, get i8* pointer
                    let temp_alloca = match self
                        .builder
                        .build_alloca(elem_val.get_type(), "push_val_alloca")
                    {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during alloca: {}", e),
                                span,
                            ))
                        }
                    };
                    self.builder.build_store(temp_alloca, elem_val);
                    let value_ptr_i8 = match self.builder.build_pointer_cast(
                        temp_alloca,
                        self.context.i8_type().ptr_type(AddressSpace::default()),
                        "push_val_ptr",
                    ) {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during pointer cast: {}", e),
                                span,
                            ))
                        }
                    };

                    // Call runtime push function
                    let vec_push_func = self.get_or_declare_vec_push();
                    self.builder.build_call(
                        vec_push_func,
                        &[vec_handle_ptr.into(), value_ptr_i8.into()],
                        "",
                    );

                    Ok(self.context.f64_type().const_float(0.0).into()) // push returns "void" (0.0 float placeholder)
                } else if name == "len" {
                    // Arg 0 is vector or string handle (i8*)
                    let handle_val = self.compile_expression(&args[0])?;
                    let handle_ptr = match handle_val {
                        BasicValueEnum::PointerValue(pv)
                            if pv.get_type()
                                == self.context.i8_type().ptr_type(AddressSpace::default()) =>
                        {
                            pv
                        }
                        _ => {
                            return Err(CodeGenError::InvalidType(
                                format!(
                                    "Expected vector or string handle (i8*), found {}",
                                    handle_val.get_type()
                                ),
                                span,
                            ))
                        }
                    };
                    // Call appropriate runtime length function
                    // Assume type checker ensured arg is vec or string
                    // We don't *actually* know here which it was without TC info!
                    // HACK: Call str_len, assumes runtime handles both or TC passed info.
                    let len_func = self.get_or_declare_str_len(); // Or vec_len
                    let len_i64 =
                        match self
                            .builder
                            .build_call(len_func, &[handle_ptr.into()], "len")
                        {
                            Ok(call) => call
                                .try_as_basic_value()
                                .left()
                                .ok_or(CodeGenError::LlvmError(
                                    "Failed to get length from call".to_string(),
                                    span,
                                ))?
                                .into_int_value(), // i64 length
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("Call to _phoenix_str_len failed: {}", e),
                                    span,
                                ))
                            }
                        };
                    Ok(len_i64.into()) // len returns Int
                }
                // --- Built-in pop ---
                else if name == "pop" {
                    // Arg 0 is vector handle (i8*)
                    let vec_handle_val = self.compile_expression(&args[0])?;
                    let vec_handle_ptr = match vec_handle_val {
                        BasicValueEnum::PointerValue(pv)
                            if pv.get_type() == self.context.ptr_type(AddressSpace::default()) =>
                        {
                            pv
                        }
                        _ => {
                            return Err(CodeGenError::InvalidType(
                                format!(
                                    "Expected vector handle (i8*), found {}",
                                    vec_handle_val.get_type()
                                ),
                                span,
                            ))
                        }
                    };

                    // Get Element Type (Needs info passed from Type Checker!)
                    let vec_expr_node = &args[0]; // The AST node for the vector expression
                    let vec_resolved_type = vec_expr_node.get_type();
                    let elem_toy_type = match vec_resolved_type {
                        Some(Type::Vector(et)) => *et,
                        _ => {
                            return Err(CodeGenError::InvalidType(
                                format!(
                                    "Expected vector type, found {}",
                                    vec_resolved_type.unwrap_or(Type::Void)
                                ),
                                span,
                            ))
                        }
                    };
                    let elem_llvm_type = elem_toy_type.to_llvm_basic_type(self.context).unwrap();

                    // Call runtime vec_pop(handle) -> i8* (pointer to popped data)
                    let vec_pop_func = self.get_or_declare_vec_pop();
                    let popped_elem_ptr_i8 = match self.builder.build_call(
                        vec_pop_func,
                        &[vec_handle_ptr.into()],
                        "pop_ptr_i8",
                    ) {
                        Ok(call) => call
                            .try_as_basic_value()
                            .left()
                            .ok_or(CodeGenError::LlvmError(
                                "Failed to get popped element pointer from call".to_string(),
                                span.clone(),
                            ))?
                            .into_pointer_value(), // i8* handle
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("Call to _phoenix_vec_pop failed: {}", e),
                                span.clone(),
                            ))
                        }
                    };
                    // TODO: Check for null return from pop (empty vector)

                    // Cast the i8* to the correct element type pointer
                    let elem_ptr_typed = match self.builder.build_pointer_cast(
                        popped_elem_ptr_i8,
                        elem_llvm_type.ptr_type(AddressSpace::default()),
                        "pop_ptr_typed",
                    ) {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during pointer cast: {}", e),
                                span,
                            ))
                        }
                    };
                    // Load the value from that pointer
                    let popped_value =
                        match self
                            .builder
                            .build_load(elem_llvm_type, elem_ptr_typed, "pop_val")
                        {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during load: {}", e),
                                    span,
                                ))
                            }
                        };

                    Ok(popped_value) // Pop returns the element value
                }
                // --- User Function Call ---
                else {
                    let (param_types, return_type, func_val) = self
                        .functions
                        .get(name)
                        .ok_or_else(|| CodeGenError::UndefinedFunction(name.clone(), span.clone()))?
                        .clone();

                    if args.len() != param_types.len() {
                        return Err(CodeGenError::IncorrectArgumentCount {
                            func_name: name.clone(),
                            expected: param_types.len(),
                            found: args.len(),
                            span,
                        });
                    }

                    let mut compiled_args: Vec<BasicMetadataValueEnum> =
                        Vec::with_capacity(args.len());
                    for (i, arg_expr) in args.iter().enumerate() {
                        let arg_val = self.compile_expression(arg_expr)?;
                        let expected_llvm_type =
                            match param_types[i].to_llvm_basic_type(self.context) {
                                Some(llvm_type) => llvm_type,
                                None => {
                                    return Err(CodeGenError::InvalidType(
                                        format!(
                                            "Invalid type for argument {} in function '{}'",
                                            i, name
                                        ),
                                        span,
                                    ))
                                }
                            };
                        // --- Type Checking/Conversion (Basic) ---
                        if arg_val.get_type() != expected_llvm_type {
                            // Attempt basic conversion (e.g., int to float) if needed later
                            // Or error out
                            return Err(CodeGenError::InvalidType(format!(
                                "Argument {} type mismatch for function '{}': expected {}, found {}",
                                i, name, expected_llvm_type, arg_val.get_type()
                            ), span));
                        }
                        compiled_args.push(arg_val.into());
                    }

                    let call_site_val =
                        match self.builder.build_call(func_val, &compiled_args, "calltmp") {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during function call: {}", e),
                                    span,
                                ))
                            }
                        };

                    // Return value needs to match expected function return type
                    match return_type {
                        Type::Void => {
                            // How to return void from an expression? Error? Special Void value?
                            // Let's return 0.0 float for now, needs refinement.
                            Ok(self.context.f64_type().const_float(0.0).into())
                        }
                        _ => {
                            // Expect Int, Float, or Bool
                            let expected_llvm_ret_type =
                                match return_type.to_llvm_basic_type(self.context) {
                                    Some(llvm_type) => llvm_type,
                                    None => {
                                        return Err(CodeGenError::InvalidType(
                                            format!("Invalid return type for function '{}'", name),
                                            span,
                                        ))
                                    }
                                };
                            match call_site_val.try_as_basic_value().left() {
                                Some(ret_val) if ret_val.get_type() == expected_llvm_ret_type => {
                                    Ok(ret_val)
                                }
                                _ => Err(CodeGenError::LlvmError(
                                    format!(
                                        "Call to '{}' did not return expected type {}",
                                        name, expected_llvm_ret_type
                                    ),
                                    span,
                                )),
                            }
                        }
                    }
                }
            } // End FunctionCall
            // --- Assignment ---
            ExpressionKind::Assignment { target, value } => {
                // println!("Assignment: {:#?}", target);
                // 1. Compile the RHS value FIRST
                let compiled_rhs_value = self.compile_expression(value)?;

                // 2. Compile the LHS target expression into a POINTER to the location
                let target_ptr = self.compile_lvalue_pointer(target)?; // Get PointerValue

                // 3. Type Check (Defensive)
                let target_llvm_ptr_type = target_ptr.get_type(); // e.g., ptr to i64
                let target_type =
                    self.check_lvalue_expression(target)
                        .ok_or(CodeGenError::InvalidType(
                            format!(
                                "Invalid target type for assignment: {}",
                                target_ptr.get_type()
                            ),
                            value.span.clone(),
                        ))?;
                let expected_rhs_llvm_type_opt = target_type.to_llvm_basic_type(self.context); // Get LLVM type from ToyLang Type
                let Some(expected_rhs_llvm_type) = expected_rhs_llvm_type_opt else {
                    // Handle case where target type (e.g., Void) doesn't map to BasicType
                    return Err(CodeGenError::InvalidType(
                        format!(
                            "Invalid target type for assignment: {}",
                            target_ptr.get_type()
                        ),
                        value.span.clone(),
                    ));
                };
                // todo fix this typecheck, because it currently brakes for nested vectors. for I just assume that the typechecker detects all errors
                // if compiled_rhs_value.get_type() != expected_rhs_llvm_type {
                //     // This indicates internal error or missing conversion
                //     return Err(CodeGenError::InvalidType(format!(
                //         "Internal Codegen Error: Assignment type mismatch: target {:?} vs value {:?}",
                //         expected_rhs_llvm_type, compiled_rhs_value.get_type()
                //     ), value.span.clone()));
                // }

                // 4. Generate the store instruction using the target pointer
                self.builder.build_store(target_ptr, compiled_rhs_value);

                // 5. Assignment expression returns the assigned value
                Ok(compiled_rhs_value)
            } // End Assignment case

            // --- If Expression (Branches are now Expressions) ---
            ExpressionKind::IfExpr {
                condition,
                then_branch,
                else_branch,
            } => {
                // Compile condition, check if bool (i1)
                let cond_val = self.compile_expression(condition)?;
                let bool_cond = match cond_val {
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            format!(
                                "If condition must be boolean (i1), found {}",
                                cond_val.get_type()
                            ),
                            span,
                        ))
                    }
                };

                // Setup Blocks
                let current_func = self
                    .current_function
                    .expect("Cannot compile 'if' expression without current function");
                let then_bb = self.context.append_basic_block(current_func, "then_expr");
                let else_bb = self.context.append_basic_block(current_func, "else_expr");
                let merge_bb = self.context.append_basic_block(current_func, "ifcont_expr");

                // Build Conditional Branch
                self.builder
                    .build_conditional_branch(bool_cond, then_bb, else_bb);

                // Compile THEN Branch Expression
                self.builder.position_at_end(then_bb);
                // Recursively compile the expression for the 'then' branch
                let then_val = self.compile_expression(then_branch)?;
                // Branch to merge block if not already terminated
                if then_bb.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(merge_bb);
                }
                let then_end_bb = self.builder.get_insert_block().unwrap_or(then_bb);

                // Compile ELSE Branch Expression
                self.builder.position_at_end(else_bb);
                // Recursively compile the expression for the 'else' branch
                let else_val = self.compile_expression(else_branch)?;
                // Branch to merge block if not already terminated
                if else_bb.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(merge_bb);
                }
                let else_end_bb = self.builder.get_insert_block().unwrap_or(else_bb);

                // Merge Block and PHI Node
                self.builder.position_at_end(merge_bb);
                if then_val.get_type() != else_val.get_type() {
                    return Err(CodeGenError::InvalidType(
                        format!(
                            "If branches must have the same type: then {}, else {}",
                            then_val.get_type(),
                            else_val.get_type()
                        ),
                        span,
                    ));
                }
                let phi_type = then_val.get_type();
                let phi_node = match self.builder.build_phi(phi_type, "ifexpr_tmp") {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("LLVM error creating PHI node: {}", e),
                            span,
                        ))
                    }
                };
                phi_node.add_incoming(&[(&then_val, then_end_bb), (&else_val, else_end_bb)]);

                Ok(phi_node.as_basic_value())
            } // End IfExpr case

            // --- Block Expression ---
            ExpressionKind::Block {
                statements,
                final_expression,
            } => {
                // --- Scoping (Important!) ---
                // Need to handle variable scopes if blocks introduce them.
                let original_vars = self.variables.clone(); // Save original scope
                self.variables = HashMap::new(); // Clear current scope

                // 1. Compile the statements sequentially
                for stmt in statements {
                    // Check if statement terminated block
                    if let Ok((_, terminated)) = self.compile_statement(stmt) {
                        if terminated {
                            // Block terminated early, what should its value be?
                            // This implies unreachable code after it in the AST block.
                            // Type checker should ideally catch this?
                            // For codegen, we probably can't produce a value if terminated.
                            // Let's return Void represented by default float 0.0 for now,
                            // but signal this isn't right.
                            eprintln!(
                                "Warning: Code block terminated early by return/break/continue."
                            );
                            return Ok(self.context.f64_type().const_float(0.0).into());
                            // Placeholder!
                        }
                    } else {
                        // Error compiling statement, propagate?
                        // Current CompileStmtResult doesn't propagate well here. Needs refactor.
                        // For now, assume errors are handled elsewhere or ignore.
                    }
                }

                // 2. Compile the final expression if it exists
                let result = if let Some(final_expr) = final_expression {
                    self.compile_expression(final_expr)? // Result of block is final expr
                } else {
                    // Block has no final expression. What should it return?
                    // For now, let's default to float 0.0. Needs refinement based on typing.
                    // Or maybe error if used where value needed? Let's try 0.0 float default.
                    self.context.f64_type().const_float(0.0).into()
                };

                // --- Restore Scope
                self.variables = original_vars;
                // Return the result of the final expression

                Ok(result)
            }

            // --- Vector Literal ---
            ExpressionKind::VectorLiteral { elements } => {
                // --- Determine Element Type (Using resolved type from AST) ---
                let vector_type = expr.get_type().unwrap_or(Type::Void); // Get vec<T> type
                let elem_toy_type = match vector_type {
                    Type::Vector(et) => *et,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            "Vector literal node missing Vector type".into(),
                            expr.span.clone(),
                        ))
                    }
                };
                if elem_toy_type == Type::Void { /* Error */ }
                // Get size of ELEMENT for vec_new, but size of HANDLE if elements are vectors
                // No, vec_new takes element size. If elem is vec<int>, size is pointer size.
                let elem_size = self.get_sizeof(&elem_toy_type).unwrap();
                let elem_llvm_type = elem_toy_type.to_llvm_basic_type(self.context).unwrap(); // Basic type of element (e.g., i8*, i64)

                // --- Call vec_new ---
                let vec_new_func = self.get_or_declare_vec_new();
                let capacity = self
                    .context
                    .i64_type()
                    .const_int(elements.len() as u64, false);
                let elem_size_val = self.context.i64_type().const_int(elem_size, false);
                let vec_handle = match self.builder.build_call(
                    vec_new_func,
                    &[elem_size_val.into(), capacity.into()],
                    "new_vec",
                ) {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("LLVM error during vec_new call: {}", e),
                            span,
                        ))
                    }
                }
                .try_as_basic_value()
                .left()
                .ok_or(CodeGenError::LlvmError(
                    "vec_new call failed".to_string(),
                    expr.span.clone(),
                ))?
                .into_pointer_value(); // i8* handle

                // --- Populate Vector using vec_push ---
                let vec_push_func = self.get_or_declare_vec_push();
                for elem_expr in elements.iter() {
                    let elem_val = self.compile_expression(elem_expr)?; // Compile element (might be i8* handle for inner vec)
                                                                        // Check type match (redundant if TC works)
                    if elem_val.get_type() != elem_llvm_type { /* Internal Error */ }

                    // Allocate temp space, store, cast pointer, call vec_push
                    let temp_alloca =
                        match self.builder.build_alloca(elem_llvm_type, "push_val_alloca") {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during alloca: {}", e),
                                    span,
                                ))
                            }
                        };
                    self.builder.build_store(temp_alloca, elem_val);
                    let value_ptr_i8 = match self.builder.build_pointer_cast(
                        temp_alloca,
                        self.context.ptr_type(AddressSpace::default()),
                        "push_val_ptr",
                    ) {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during pointer cast: {}", e),
                                span,
                            ))
                        }
                    };
                    self.builder.build_call(
                        vec_push_func,
                        &[vec_handle.into(), value_ptr_i8.into()],
                        "",
                    );
                }
                Ok(vec_handle.into()) // Return outer vector handle
            }

            // --- Index Access ---
            ExpressionKind::IndexAccess { target, index } => {
                // This logic computes the pointer *and then loads*.
                // We can reuse the pointer calculation logic.
                let elem_ptr_typed = self.compile_lvalue_pointer(expr)?; // Treat IndexAccess itself as an L-Value to get pointer

                // Load the value from the element pointer
                // Need the element LLVM type again here
                let target_resolved_type = target.get_type();
                let elem_toy_type = match target_resolved_type {
                    Some(Type::Vector(et)) => *et,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            format!(
                                "Index access target must be a vector, found {}",
                                target_resolved_type.unwrap_or(Type::Void)
                            ),
                            span,
                        ))
                    }
                };
                let elem_llvm_type = elem_toy_type.to_llvm_basic_type(self.context).unwrap();

                let loaded_val =
                    match self
                        .builder
                        .build_load(elem_llvm_type, elem_ptr_typed, "load_idx")
                    {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during load: {}", e),
                                span,
                            ))
                        }
                    };
                Ok(loaded_val)
            }
        } // End match expr
    }

    // Ensure check_lvalue_expression returns the correct ToyLang Type
    fn check_lvalue_expression(&mut self, target: &Expression) -> Option<Type> {
        // Check if target is a variable
        if let ExpressionKind::Variable(var_name) = &target.kind {
            // Check if variable exists in current scope
            if let Some(var_type) = self.variables.get(var_name) {
                return Some(var_type.ty.clone());
            }
        }
        // Check if target is an index access
        if let ExpressionKind::IndexAccess { target, index } = &target.kind {
            // Check if target is a vector
            if let Some(Type::Vector(_)) = target.get_type() {
                return Some(target.get_type().unwrap());
            }
        }
        None // Not a valid lvalue
    }

    // Helper to compile a block (Program) and return the value of its last ExpressionStmt
    // Returns Ok(None) if block is empty or has no final ExpressionStmt
    // Helper to compile blocks (Program), returns bool indicating if block terminated itself
    fn compile_program_block(&mut self, program: &Program) -> Result<bool, CodeGenError> {
        // Changed return type
        let mut terminated = false;
        for stmt in &program.statements {
            let (stmt_val_opt, stmt_terminated) = self.compile_statement(stmt)?;
            // We ignore stmt_val_opt here as this block isn't an expression block
            if stmt_terminated {
                terminated = true;
                break; // Stop processing statements after a terminator
            }
        }
        Ok(terminated)
    }

    fn get_or_declare_print_float_wrapper(&mut self) -> Result<FunctionValue<'ctx>, CodeGenError> {
        // Check cache... (Assume Compiler struct has print_float_wrapper_func: Option<...>)
        if let Some(f) = self.print_float_wrapper_func {
            return Ok(f);
        }
        let f64_type = self.context.f64_type();
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[f64_type.into()], false);
        let func = self
            .module
            .add_function("print_f64_wrapper", fn_type, Some(Linkage::External));
        self.print_float_wrapper_func = Some(func);
        Ok(func)
    }

    fn get_or_declare_print_int_wrapper(&mut self) -> Result<FunctionValue<'ctx>, CodeGenError> {
        if let Some(f) = self.print_int_wrapper_func {
            return Ok(f);
        }
        // Check cache... (Assume print_int_wrapper_func field)
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[i64_type.into()], false);
        let func = self
            .module
            .add_function("print_i64_wrapper", fn_type, Some(Linkage::External));
        self.print_int_wrapper_func = Some(func);
        Ok(func)
    }

    fn get_or_declare_print_bool_wrapper(&mut self) -> Result<FunctionValue<'ctx>, CodeGenError> {
        // Check cache... (Assume print_bool_wrapper_func field)
        if let Some(f) = self.print_bool_wrapper_func {
            return Ok(f);
        }
        let i1_type = self.context.bool_type(); // bool_type() is i1
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[i1_type.into()], false);
        let func = self
            .module
            .add_function("print_bool_wrapper", fn_type, Some(Linkage::External));
        self.print_bool_wrapper_func = Some(func);
        Ok(func)
    }

    fn get_or_declare_print_str_wrapper(&mut self) -> Result<FunctionValue<'ctx>, CodeGenError> {
        if let Some(f) = self.print_str_wrapper_func {
            return Ok(f);
        } // Check cache

        // Signature: void print_str_wrapper(i8*)
        let i8_ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[i8_ptr_type.into()], false);

        let func = self
            .module
            .add_function("print_str_wrapper", fn_type, Some(Linkage::External));
        self.print_str_wrapper_func = Some(func); // Store in cache
        Ok(func)
    }

    // Helper to get or declare the print_str_ln wrapper function
    fn get_or_declare_print_str_ln_wrapper(&mut self) -> Result<FunctionValue<'ctx>, CodeGenError> {
        if let Some(f) = self.print_str_ln_wrapper_func {
            return Ok(f);
        } // Check cache

        // Signature: void print_str_ln_wrapper_func()
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[], false);

        let func = self.module.add_function(
            "print_str_ln_wrapper_func",
            fn_type,
            Some(Linkage::External),
        );
        self.print_str_ln_wrapper_func = Some(func); // Store in cache
        Ok(func)
    }

    // Declare helper for vec_new
    fn get_or_declare_vec_new(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.vec_new_func {
            return f;
        }
        let i64_t = self.context.i64_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        // takes elem_size (i64), capacity (i64), returns handle (i8*)
        let fn_type = i8ptr_t.fn_type(&[i64_t.into(), i64_t.into()], false);
        let func = self
            .module
            .add_function(RUNTIME_VEC_NEW, fn_type, Some(Linkage::External));
        self.vec_new_func = Some(func);
        func
    }
    // Declare helper for vec_get_ptr
    fn get_or_declare_vec_get_ptr(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.vec_get_ptr_func {
            return f;
        }
        let i64_t = self.context.i64_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        // takes handle (i8*), index (i64), returns element ptr (i8*)
        let fn_type = i8ptr_t.fn_type(&[i8ptr_t.into(), i64_t.into()], false);
        let func = self
            .module
            .add_function(RUNTIME_VEC_GET_PTR, fn_type, Some(Linkage::External));
        self.vec_get_ptr_func = Some(func);
        func
    }

    fn get_or_declare_vec_len(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.vec_len_func {
            return f;
        }
        let i64_t = self.context.i64_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = i64_t.fn_type(&[i8ptr_t.into()], false); // i8* -> i64
        let func = self
            .module
            .add_function(RUNTIME_VEC_LEN, fn_type, Some(Linkage::External));
        self.vec_len_func = Some(func);
        func
    }
    fn get_or_declare_vec_push(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.vec_push_func {
            return f;
        }
        let void_t = self.context.void_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        // takes handle (i8*), value_ptr (i8*) -> void
        let fn_type = void_t.fn_type(&[i8ptr_t.into(), i8ptr_t.into()], false);
        let func = self
            .module
            .add_function(RUNTIME_VEC_PUSH, fn_type, Some(Linkage::External));
        self.vec_push_func = Some(func);
        func
    }
    fn get_or_declare_vec_free(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.vec_free_func {
            return f;
        }
        let void_t = self.context.void_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = void_t.fn_type(&[i8ptr_t.into()], false); // i8* -> void
        let func = self
            .module
            .add_function(RUNTIME_VEC_FREE, fn_type, Some(Linkage::External));
        self.vec_free_func = Some(func);
        func
    }

    fn get_or_declare_vec_pop(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.vec_pop_func {
            return f;
        }
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = i8ptr_t.fn_type(&[i8ptr_t.into()], false); // i8* -> i8*
        let func = self
            .module
            .add_function(RUNTIME_VEC_POP, fn_type, Some(Linkage::External));
        self.vec_pop_func = Some(func);
        func
    }
    fn get_or_declare_str_new(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.str_new_func {
            return f;
        }
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = i8ptr_t.fn_type(&[i8ptr_t.into()], false); // i8* -> i8* handle
        let func = self
            .module
            .add_function(RUNTIME_STR_NEW, fn_type, Some(Linkage::External));
        self.str_new_func = Some(func);
        func
    }
    fn get_or_declare_str_len(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.str_len_func {
            return f;
        }
        let i64_t = self.context.i64_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = i64_t.fn_type(&[i8ptr_t.into()], false); // i8* -> i64
        let func = self
            .module
            .add_function(RUNTIME_STR_LEN, fn_type, Some(Linkage::External));
        self.str_len_func = Some(func);
        func
    }
    fn get_or_declare_str_concat(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.str_concat_func {
            return f;
        }
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = i8ptr_t.fn_type(&[i8ptr_t.into(), i8ptr_t.into()], false); // i8*, i8* -> i8* handle
        let func = self
            .module
            .add_function(RUNTIME_STR_CONCAT, fn_type, Some(Linkage::External));
        self.str_concat_func = Some(func);
        func
    }

    fn get_or_declare_str_free(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.str_free_func {
            return f;
        }
        let void_t = self.context.void_type();
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = void_t.fn_type(&[i8ptr_t.into()], false); // i8* -> void
        let func = self
            .module
            .add_function(RUNTIME_STR_FREE, fn_type, Some(Linkage::External));
        self.str_free_func = Some(func);
        func
    }

    // RUNTIME_STR_GET_CHAR_PTR
    fn get_or_declare_str_get_char_ptr(&mut self) -> FunctionValue<'ctx> {
        if let Some(f) = self.str_get_char_ptr_func {
            return f;
        }
        let i8ptr_t = self.context.i8_type().ptr_type(AddressSpace::default());
        let fn_type = i8ptr_t.fn_type(&[i8ptr_t.into()], false); // i8* -> i8*
        let func =
            self.module
                .add_function(RUNTIME_STR_GET_CHAR_PTR, fn_type, Some(Linkage::External));
        self.str_get_char_ptr_func = Some(func);
        func
    }

    // Get size of a phoenix Type in bytes (basic implementation)
    // Get size of a ToyLang Type in bytes
    fn get_sizeof(&self, ty: &Type) -> Option<u64> {
        // Needs target machine data layout! Hardcoding 64-bit pointer size.
        // todo: Use target machine data layout for pointer size
        let pointer_size = 8;
        match ty {
            Type::Int | Type::Float => Some(8),
            Type::Bool => Some(1),
            // String and Vector are represented by handles (pointers)
            Type::String => Some(pointer_size),
            Type::Vector(_) => Some(pointer_size), // <<< Size of the handle/pointer
            Type::Void => Some(0),
        }
    }

    /// Compiles a single Statement node.
    /// Returns Ok(Some(FloatValue)) if it's an ExpressionStmt, Ok(None) for LetBinding, Err on failure.
    /// Compiles a single Statement node. Now runs FPM on function definitions.
    // --- Compile Statement (handles types) ---
    pub(crate) fn compile_statement(&mut self, stmt: &Statement) -> CompileStmtResult<'ctx> {
        let span = stmt.span.clone();
        match &stmt.kind {
            StatementKind::LetBinding {
                name,
                type_ann,
                value,
            } => {
                // Call helper with the determined type
                self.compile_var_let_stmt(name, value, false, span, type_ann)
            }

            StatementKind::VarBinding {
                name,
                type_ann,
                value,
            } => {
                // Call helper with the determined type
                self.compile_var_let_stmt(name, value, true, span, type_ann)
            }

            // --- Return Statement ---
            StatementKind::ReturnStmt { value } => {
                match value {
                    // Case 1: return <expr>;
                    Some(expr) => {
                        // Compile the return value expression
                        let compiled_value = self.compile_expression(expr)?;
                        // Generate `ret <type> <value>`
                        self.builder.build_return(Some(&compiled_value));
                    }
                    // Case 2: return; (for void functions)
                    None => {
                        // Generate `ret void`
                        self.builder.build_return(None);
                    }
                }
                // Return statement yields no value *to the next statement*
                // and terminates the block. The Option<BasicValue> return here
                // signifies the value of the *last expression checked within this statement*,
                // which is None if we just returned.
                // However, compile_block / function compilation handles the actual return value.
                // Let's return None to signify no value passed to subsequent stmts in sequence.
                Ok((None, true))

                // Alternative: Could return a special marker? But None seems okay.
                // The block terminator check in loops/if should handle this.
            } // End ReturnStmt

            StatementKind::ExpressionStmt(expr) => {
                // Compile the expression, result might be any basic type
                let value = self.compile_expression(expr)?;
                Ok((Some(value), false))
            }

            // --- While Statement ---
            StatementKind::WhileStmt { condition, body } => {
                let current_func = self
                    .current_function
                    .expect("Cannot compile 'while' outside a function");

                // 1. Create Basic Blocks
                // Block to evaluate the condition
                let cond_bb = self.context.append_basic_block(current_func, "while_cond");
                // Block for the loop body
                let loop_bb = self.context.append_basic_block(current_func, "while_body");
                // Block for code after the loop
                let after_bb = self.context.append_basic_block(current_func, "after_while");

                // 2. Branch from current block to condition check
                self.builder.build_unconditional_branch(cond_bb);

                // 3. Compile Condition Check Block
                self.builder.position_at_end(cond_bb);
                let cond_val = self.compile_expression(condition)?;
                let bool_cond = match cond_val {
                    // Verify condition is boolean (i1)
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            format!(
                                "While condition must be boolean (i1), found {}",
                                cond_val.get_type()
                            ),
                            span,
                        ))
                    }
                };
                // Build conditional branch based on condition value
                self.builder
                    .build_conditional_branch(bool_cond, loop_bb, after_bb);

                // 4. Compile Loop Body Block
                self.builder.position_at_end(loop_bb);
                let body_terminated = self.compile_program_block(body)?; // Compile the body statements, ignore value
                                                                         // After body, unconditionally branch back to condition check
                if !body_terminated {
                    // Only branch if block wasn't terminated (e.g. by future return/break)
                    self.builder.build_unconditional_branch(cond_bb);
                }

                // 5. Position builder at the block after the loop
                self.builder.position_at_end(after_bb);

                // While statement yields no value
                Ok((None, false))
            } // End WhileStmt

            // --- For Statement ---
            StatementKind::ForStmt {
                initializer,
                condition,
                increment,
                body,
            } => {
                let current_func = self
                    .current_function
                    .expect("Cannot compile 'for' outside a function");

                // --- Scoping ---
                // Does the initializer introduce variables visible only in the loop?
                // Our current initializer is just an expr, so no new scope needed *yet*.
                // If we allowed `for (var i=0;...)`, we'd need enter_scope/exit_scope here.
                // let original_vars = self.variables.clone();
                // self.symbol_table.enter_scope(); // If initializer creates scope

                // 1. Compile Initializer (if present)
                if let Some(init_expr) = initializer {
                    // Compile for side effects, ignore value
                    let _ = self.compile_statement(init_expr)?;
                }

                // 2. Create Basic Blocks
                let cond_bb = self.context.append_basic_block(current_func, "for_cond");
                let loop_bb = self.context.append_basic_block(current_func, "for_body");
                // Create increment block *before* after block for clearer flow graph
                let incr_bb = self.context.append_basic_block(current_func, "for_incr");
                let after_bb = self.context.append_basic_block(current_func, "after_for");

                // 3. Branch from current block to condition check
                self.builder.build_unconditional_branch(cond_bb);

                // 4. Compile Condition Check Block
                self.builder.position_at_end(cond_bb);
                let bool_cond = match condition {
                    Some(cond_expr) => {
                        // Compile condition expression, check if bool (i1)
                        let cond_val = self.compile_expression(cond_expr)?;
                        match cond_val {
                            BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => {
                                iv
                            }
                            _ => {
                                return Err(CodeGenError::InvalidType(
                                    format!(
                                        "For loop condition must be boolean (i1), found {}",
                                        cond_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        }
                    }
                    None => {
                        // No condition means loop indefinitely (or until break/return later)
                        // Generate `true` constant
                        self.context.bool_type().const_int(1, false)
                    }
                };
                // Branch conditionally based on condition
                self.builder
                    .build_conditional_branch(bool_cond, loop_bb, after_bb);

                // 5. Compile Loop Body Block
                self.builder.position_at_end(loop_bb);
                let body_terminated = self.compile_program_block(body)?; // Compile body statements
                                                                         // After body, branch to increment block (if not already terminated)
                                                                         // if loop_bb.get_terminator().is_none() {
                                                                         //     self.builder.build_unconditional_branch(incr_bb);
                                                                         // }
                if !body_terminated {
                    // goes back to increment block, only if not terminated
                    self.builder.build_unconditional_branch(incr_bb); // <<< Terminates that block
                }

                // 6. Compile Increment Block
                self.builder.position_at_end(incr_bb);
                if let Some(incr_expr) = increment {
                    // Compile increment expression for side effects, ignore value
                    let _ = self.compile_expression(incr_expr)?;
                }
                // After increment, unconditionally branch back to condition check
                // (only if block wasn't terminated by increment itself, unlikely but possible)
                if incr_bb.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(cond_bb);
                }

                // 7. Position builder at the block after the loop
                self.builder.position_at_end(after_bb);

                // --- Restore Scope (if needed) ---
                // self.symbol_table.exit_scope();
                // self.variables = original_vars;

                Ok((None, false)) // For statement yields no value
            } // End ForStmt

            // --- If Statement (optional else, no return value/PHI needed) ---
            StatementKind::IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                // Compile condition, check bool...
                let cond_val = self.compile_expression(condition)?;
                let bool_cond = match cond_val {
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            format!(
                                "If condition must be boolean (i1), found {}",
                                cond_val.get_type()
                            ),
                            span,
                        ))
                    }
                };
                let current_func = self
                    .current_function
                    .expect("Cannot compile 'if' statement without current function");
                let then_bb = self.context.append_basic_block(current_func, "then_stmt");
                let merge_bb = self.context.append_basic_block(current_func, "ifcont_stmt");
                let else_bb = if else_branch.is_some() {
                    self.context.append_basic_block(current_func, "else_stmt")
                } else {
                    merge_bb
                };
                self.builder
                    .build_conditional_branch(bool_cond, then_bb, else_bb);

                // Compile THEN branch (Program block)
                self.builder.position_at_end(then_bb);
                let body_terminated = self.compile_program_block(then_branch)?; // Compile the Program block
                if !body_terminated {
                    self.builder.build_unconditional_branch(merge_bb);
                }

                // Compile ELSE branch (Program block)
                if let Some(else_prog) = else_branch {
                    self.builder.position_at_end(else_bb);
                    let body_terminated = self.compile_program_block(else_prog)?; // Compile the Program block
                    if !body_terminated {
                        self.builder.build_unconditional_branch(merge_bb);
                    }
                }

                self.builder.position_at_end(merge_bb);
                Ok((None, false))
            }

            StatementKind::FunctionDef {
                name,
                params,
                return_type_ann,
                body,
            } => {
                if self.functions.contains_key(name) {
                    // Function already defined
                    return Err(CodeGenError::FunctionRedefinition(name.clone(), span));
                }

                // --- Determine Param and Return Types ---
                // Use annotations, default to float for now if missing (NEEDS TYPE CHECKING)
                // --- Determine Param and Return Types ---
                let toy_param_types: Vec<Type> = params
                    .iter()
                    .map(|param| {
                        // param.type_ann is Option<Type>
                        // convert type_ann: TypeNode to Type with type_node_to_type
                        match &param.1 {
                            Some(type_node) => Some(type_node_to_type(type_node)),
                            None => None, // Default to Float if no type annotation
                        }
                        .unwrap_or(Type::Void)
                    })
                    .collect();
                // convert type_ann: TypeNode to Type with type_node_to_type
                let type_ann = match return_type_ann {
                    Some(type_node) => Some(type_node_to_type(type_node)),
                    None => None, // Default to Float if no type annotation
                };
                let toy_return_type = type_ann.unwrap_or(Type::Void); // Default Float (improve later)

                let llvm_param_types: Result<Vec<BasicMetadataTypeEnum>, CodeGenError> =
                    toy_param_types
                        .iter()
                        .map(|ty| match ty.to_llvm_basic_type(self.context) {
                            Some(llvm_type) => Ok(llvm_type.into()),
                            None => Err(CodeGenError::InvalidType(
                                "Unsupported parameter type".to_string(),
                                span.clone(),
                            )),
                        })
                        .collect(); // Collects into Result<Vec<_>, _>

                let llvm_param_types = llvm_param_types?; // Propagate error if any
                let fn_type = match toy_return_type {
                    Type::Float => self.context.f64_type().fn_type(&llvm_param_types, false),
                    Type::Int => self.context.i64_type().fn_type(&llvm_param_types, false),
                    Type::Bool => self.context.bool_type().fn_type(&llvm_param_types, false),
                    Type::Void => self.context.void_type().fn_type(&llvm_param_types, false),
                    Type::String => {
                        // String is represented as a pointer (i8*)
                        self.context
                            .i8_type()
                            .ptr_type(AddressSpace::default())
                            .fn_type(&llvm_param_types, false)
                    } // Add other types like Pointer, Array, Struct later if needed
                    Type::Vector(_) => {
                        // Vector is represented as a pointer (i8*)
                        self.context
                            .ptr_type(AddressSpace::default())
                            .fn_type(&llvm_param_types, false)
                    }
                };

                let function = self.module.add_function(name, fn_type, None);
                // Store signature with FunctionValue
                self.functions.insert(
                    name.clone(),
                    (toy_param_types.clone(), toy_return_type.clone(), function),
                );

                // --- Setup Function Body Context ---
                let entry_block = self.context.append_basic_block(function, "entry");
                let original_builder_pos = self.builder.get_insert_block();
                let original_func = self.current_function;
                let original_vars = self.variables.clone();
                let original_ret_type = self.current_function_return_type.clone(); // Save outer return type

                self.builder.position_at_end(entry_block);
                self.current_function = Some(function);
                self.current_function_return_type = Some(toy_return_type.clone()); // Set expected return type
                self.variables.clear();

                // --- Allocate and Store Parameters (using determined types) ---
                for (i, (param_name, _)) in params.iter().enumerate() {
                    let param_toy_type = toy_param_types[i].clone(); // Get the type we determined
                    let llvm_param = function.get_nth_param(i as u32).unwrap();
                    llvm_param.set_name(param_name);
                    // llvm_param should already have the correct LLVM type from fn_type

                    let param_alloca =
                        self.create_entry_block_alloca(param_name, param_toy_type.clone());
                    self.builder.build_store(param_alloca, llvm_param);
                    self.variables.insert(
                        param_name.clone(),
                        VariableInfo {
                            ptr: param_alloca,
                            ty: param_toy_type,
                            is_mutable: false,
                        },
                    );
                }

                let signature = self.functions.get(name).cloned().unwrap(); // Assume defined

                // Compile the body statements, check if body guaranteed termination
                let mut body_terminated = false;
                let mut body_compile_error = None;
                match self.compile_program_block(body) {
                    // Use helper, returns Result<bool>
                    Ok(terminated) => body_terminated = terminated,
                    Err(e) => body_compile_error = Some(e), // Capture error
                }

                // Handle compilation errors first
                if let Some(err) = body_compile_error {
                    // Handle function compilation error
                    return Err(err);
                }

                // Check if Block Explicitly Returned implicitly via its structure
                // --- Check if Block Needs Return ---
                let needs_explicit_return_syntactically = self
                    .builder
                    .get_insert_block()
                    .map_or(true, |bb| bb.get_terminator().is_none());

                if needs_explicit_return_syntactically {
                    // Function reached end without terminator
                    match signature.1 {
                        // signature.1 is return type
                        Type::Void => {
                            // OK to implicitly return void
                            self.builder.build_return(None);
                        }
                        non_void_type => {
                            // ERROR: Non-void function reached end without return.
                            // Type checker *should* have caught this if path analysis existed.
                            // Codegen cannot proceed meaningfully.
                            // We could insert 'unreachable' or just let verification fail.
                            // Let's let verification fail for now by not adding a terminator.
                            eprintln!("Codegen Error: Non-void function '{}' ({}) reached end without returning a value.", name, non_void_type);
                            // No build_return here! Verification will fail.
                        }
                    }
                }

                // --- Restore Outer Context ---
                self.variables = original_vars;
                self.current_function = original_func;
                self.current_function_return_type = original_ret_type; // Restore outer return type
                if let Some(bb) = original_builder_pos {
                    self.builder.position_at_end(bb);
                }

                // --- Handle errors / Verification / Optimization ---
                // ... (similar logic, run FPM, check verification) ...
                if body_compile_error.is_some() || !function.verify(true) {
                    // Handle function verification failure
                    if let Some(err) = body_compile_error {
                        return Err(err);
                    }
                    return Err(CodeGenError::LlvmError(
                        "Function verification failed".to_string(),
                        span,
                    ));
                }
                self.fpm.run_on(&function);
                if !function.verify(true) {
                    return Err(CodeGenError::LlvmError(
                        "Function verification failed after optimization".to_string(),
                        span,
                    ));
                }

                Ok((None, false))
            } // End FunctionDef
        } // End match stmt
    }

    fn compile_var_let_stmt(
        &mut self,
        name: &String,
        value: &Expression,
        is_mutable: bool,
        span: Span,
        type_ann: &Option<TypeNode>,
    ) -> CompileStmtResult<'ctx> {
        // Compile the value expression first
        let compiled_value = self.compile_expression(value)?;

        // --- Determine Variable Type ---
        // Use the type resolved by the type checker stored on the value expression node
        let var_type = match value.get_type() {
            // Use helper to get Option<Type>
            Some(ty) => {
                // Type checker ran successfully, use the resolved type.
                // Optional: Double-check against type_ann if it exists?
                if let Some(ann_node) = type_ann {
                    // Resolve annotation node again (or better, pass resolved Type from TC)
                    if let Some(ann_type) = resolve_type_node(ann_node, &mut vec![]) {
                        // Dummy errors vec
                        if ty != ann_type {
                            // This signals an internal inconsistency if TC passed!
                            return Err(CodeGenError::InvalidType(
                                format!("Internal Error: Inferred value type {} mismatches annotation {} for '{}'", ty, ann_type, name),
                                span
                            ));
                        }
                    } else { /* Error resolving annotation - should be caught by TC */
                    }
                }
                ty // Use the type from the value expression
            }
            None => {
                // This means the type checker failed to resolve the value's type.
                // This shouldn't happen if the type checker ran successfully first.
                return Err(CodeGenError::InvalidType(
                    format!(
                        "Internal Error: Could not determine type for RHS of binding '{}'",
                        name
                    ),
                    value.span.clone(), // Use value's span
                ));
            }
        };
        // --- End Determine Variable Type ---

        // Ensure we are not binding void
        if var_type == Type::Void {
            return Err(CodeGenError::InvalidType(
                format!("Cannot declare variable '{}' with void type", name),
                span, // Use statement span
            ));
        }
        // --- Type Check (Codegen sanity check) ---
        // The type checker should ensure this, but a check here is defensive.
        let expected_llvm_type = match var_type.to_llvm_basic_type(self.context) {
            Some(t) => t,
            None if var_type == Type::Void => {
                // This error should have been caught by type checker (VoidAssignment)
                return Err(CodeGenError::InvalidType(
                    "Cannot create variable of type void".to_string(),
                    span,
                ));
            }
            None => {
                // Should not happen for valid storable types
                return Err(CodeGenError::InvalidType(
                    format!("Cannot get basic LLVM type for variable type {}", var_type),
                    span,
                ));
            }
        };

        if compiled_value.get_type() != expected_llvm_type {
            // This indicates an internal error OR missing type conversion step
            return Err(CodeGenError::InvalidType(
                format!(
                    "Internal Error: RHS value type {} does not match variable type {} for '{}'",
                    compiled_value.get_type(),
                    expected_llvm_type,
                    name
                ),
                span,
            ));
        }
        // --- End Type Check ---

        // Allocate based on the determined type passed in
        let alloca = self.create_entry_block_alloca(name, var_type.clone());
        self.builder.build_store(alloca, compiled_value);

        // Store type info along with pointer
        self.variables.insert(
            name.clone(),
            VariableInfo {
                ptr: alloca,
                ty: var_type, // Use the passed-in type
                is_mutable,
            },
        );
        Ok((None, false)) // Binding statement yields no value, doesn't terminate block
    }

    // End compile_statement

    /// Emits the compiled module to an object file.
    /// Should be called *after* compile_program_to_module.
    pub fn emit_object_file(&self, output_path: &Path) -> Result<(), CodeGenError> {
        // --- Target Configuration ---
        // Determine Target Triple for M4 Pro Mac
        let target_triple = &TargetTriple::create("aarch64-apple-darwin"); // ARM64 macOS

        // Initialize required targets (native is usually sufficient for host compilation)
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| {
            CodeGenError::LlvmError(
                format!("Failed to initialize native target: {}", e),
                Span::default(),
            )
        })?;
        // Alternatively, initialize specific targets: Target::initialize_aarch64(&InitializationConfig::default());

        // --- Target Lookup ---
        let target = Target::from_triple(target_triple).map_err(|e| {
            CodeGenError::LlvmError(
                format!("Failed to get target for triple '{}': {}", target_triple, e),
                Span::default(),
            )
        })?;

        // --- Create Target Machine ---
        // Configure CPU, features, optimization level etc.
        let cpu = "generic"; // Be specific for M4 - "generic" or "" might also work
        let features = ""; // Use "" for default features for the CPU
        let opt_level = OptimizationLevel::Default; // Or match FPM level used
        let reloc_mode = RelocMode::Default; // Position Independent Code (PIC) is common default
        let code_model = CodeModel::Default; // Small/Kernel/Medium/Large - Default is usually fine

        let target_machine = target
            .create_target_machine(
                target_triple,
                cpu,
                features,
                opt_level,
                reloc_mode,
                code_model,
            )
            .ok_or_else(|| {
                CodeGenError::LlvmError(
                    format!(
                        "Failed to create target machine for triple '{}'",
                        target_triple
                    ),
                    Span::default(),
                )
            })?;

        // --- Emit Object File ---
        println!("Emitting object file to: {}", output_path.display());
        target_machine
            .write_to_file(
                &self.module,     // The LLVM module containing the compiled code
                FileType::Object, // Specify we want an object file
                output_path,
            )
            .map_err(|e| {
                CodeGenError::LlvmError(
                    format!("Failed to write object file: {}", e),
                    Span::default(),
                )
            })?;

        Ok(())
    }
}
// Need resolve_type_node helper accessible here or pass resolved type differently
// Placeholder - assumes resolve_type_node is available or move logic
pub fn resolve_type_node(node: &TypeNode, _errors: &mut Vec<CodeGenError>) -> Option<Type> {
    // Minimal version for codegen context - real logic is in typechecker
    match &node.kind {
        TypeNodeKind::Simple(name) => match name.as_str() {
            "int" => Some(Type::Int),
            "float" => Some(Type::Float),
            "bool" => Some(Type::Bool),
            "string" => Some(Type::String),
            "void" => Some(Type::Void),
            _ => None,
        },
        TypeNodeKind::Vector(et_node) => {
            resolve_type_node(et_node, _errors).map(|t| Type::Vector(Box::new(t)))
        }
    }
}

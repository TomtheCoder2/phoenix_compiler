// src/codegen.rs

use crate::ast::{
    type_node_to_type, BinaryOperator, ComparisonOperator, Expression, ExpressionKind, Program,
    Statement, StatementKind, TypeNode, TypeNodeKind, UnaryOperator,
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
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::AnyValue;
use inkwell::values::{BasicMetadataValueEnum, BasicValue};
use inkwell::values::{BasicValueEnum, FunctionValue, GlobalValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use std::collections::HashMap;
use std::default::Default;
use std::ffi::CString;
use std::fmt;
use std::path::Path;

const RUNTIME_VEC_NEW: &str = "_toylang_vec_new"; // fn(elem_size: i64, capacity: i64) -> i8*
const RUNTIME_VEC_GET_PTR: &str = "_toylang_vec_get_ptr"; // fn(vec_handle: i8*, index: i64) -> i8* (pointer to element)
const RUNTIME_VEC_LEN: &str = "_toylang_vec_len"; // fn(vec_handle: i8*) -> i64
const RUNTIME_VEC_PUSH: &str = "_toylang_vec_push"; // fn(vec_handle: i8*, value_ptr: i8*) -> void
const RUNTIME_VEC_FREE: &str = "_toylang_vec_free"; // fn(vec_handle: i8*) -> void

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
        }
    }
}
// Result now often returns BasicValueEnum as expressions can yield int, float, or bool
type CompileExprResult<'ctx> = Result<BasicValueEnum<'ctx>, CodeGenError>;
// CompileStmtResult remains similar, might yield BasicValueEnum if expr stmt returns non-float
type CompileStmtResult<'ctx> = Result<(Option<BasicValueEnum<'ctx>>, bool), CodeGenError>;

// Helper struct to store variable info (pointer + type)
#[derive(Debug, Clone)] // Copy is efficient as PointerValue/Type are Copy
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
    // Alloca helper now needs the ToyLang Type to allocate correctly
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
        // Allocate memory for the correct LLVM type based on ToyLang Type
        let llvm_type = match ty.to_llvm_basic_type(self.context) {
            Some(llvm_type) => llvm_type,
            None => {
                panic!("Invalid type for variable '{}'", name);
            }
        };
        // todo: check if unwrap is safe
        temp_builder.build_alloca(llvm_type, name).unwrap()
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
                // 1. Create a global constant for the string literal's value
                //    Use a unique name based on value? Or let LLVM handle duplicates?
                //    Let's use a simple name for now. Need a better naming scheme later.
                let global_name = format!("g_str_{}", self.module.get_globals().count()); // Simple unique name
                let global_val = self.create_global_string(&global_name, value, Linkage::Private);

                // 2. Get the i8* pointer to the global string constant
                let zero_idx = self.context.i32_type().const_int(0, false);
                let string_type = global_val.get_value_type().into_array_type();
                let ptr_val = match unsafe {
                    self.builder.build_in_bounds_gep(
                        string_type,
                        global_val.as_pointer_value(),
                        &[zero_idx, zero_idx],
                        "str_ptr",
                    )
                } {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("LLVM error during string pointer generation: {}", e),
                            span,
                        ))
                    }
                };

                // 3. A StringLiteral expression evaluates to the i8* pointer
                Ok(ptr_val.into()) // Return PointerValue wrapped in BasicValueEnum
            }

            // Arithmetic Operations (Example: requires operands to be the same type for now)
            ExpressionKind::BinaryOp { op, left, right } => {
                let lhs = self.compile_expression(left)?;
                let rhs = self.compile_expression(right)?;

                // Basic type checking (Replace with proper type checker later)
                match (lhs, rhs) {
                    (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                        let result = match match op {
                            BinaryOperator::Add => self.builder.build_int_add(l, r, "addtmp"),
                            BinaryOperator::Subtract => self.builder.build_int_sub(l, r, "subtmp"),
                            BinaryOperator::Multiply => self.builder.build_int_mul(l, r, "multmp"),
                            BinaryOperator::Divide => self.builder.build_int_signed_div(l, r, "sdivtmp"), // Signed division
                        } {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(format!(
                                    "LLVM error during integer operation: {}",
                                    e
                                ), span))
                            }
                        };
                        Ok(result.into())
                    }
                    (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                        let result = match match op {
                            BinaryOperator::Add => self.builder.build_float_add(l, r, "faddtmp"),
                            BinaryOperator::Subtract => self.builder.build_float_sub(l, r, "fsubtmp"),
                            BinaryOperator::Multiply => self.builder.build_float_mul(l, r, "fmultmp"),
                            BinaryOperator::Divide => self.builder.build_float_div(l, r, "fdivtmp"),
                        } {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(format!(
                                    "LLVM error during float operation: {}",
                                    e
                                ), span))
                            }
                        };
                        Ok(result.into())
                    }
                    _ => Err(CodeGenError::InvalidBinaryOperation(
                        format!("Type mismatch or unsupported types for operator {:?} (lhs: {:?}, rhs: {:?})", op, lhs.get_type(), rhs.get_type())
                        , span))
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
                                    format!("Unsupported comparison {:?} for booleans", op),
                                    span,
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(CodeGenError::InvalidBinaryOperation(
                            format!(
                                "Type mismatch for comparison operator {:?} (lhs: {:?}, rhs: {:?})",
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
                                    "Cannot apply arithmetic negate '-' to type {:?}",
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
                                    "Cannot apply logical not '!' to type {:?}",
                                    compiled_operand.get_type()
                                ),
                                span,
                            )),
                        }
                    }
                } // End match op
            }
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
                                        "print_int expects an integer argument, found {:?}",
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
                                        "Argument must be a string literal (evaluating to i8*), found {:?}",
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
                            == self.context.i8_type().ptr_type(AddressSpace::default()) => {
                            pv
                        }
                        _ => {
                            return Err(CodeGenError::InvalidType(
                                format!(
                                    "Expected vector handle (i8*), found {:?}",
                                    vec_handle_val.get_type()
                                ),
                                span,
                            ))
                        }
                    };

                    // Allocate temp space for element, store it, get i8* pointer
                    let temp_alloca = match self
                        .builder
                        .build_alloca(elem_val.get_type(), "push_val_alloca"){
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
                    ){
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
                                "Argument {} type mismatch for function '{}': expected {:?}, found {:?}",
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
                                        "Call to '{}' did not return expected type {:?}",
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
                // Compile RHS first
                let compiled_rhs_value = self.compile_expression(value)?;

                // --- Generate code based on L-value target ---
                match &target.kind {
                    // Case 1: Variable Assignment: target is Variable(name)
                    ExpressionKind::Variable(name) => {
                        let var_info = self.variables.get(name).ok_or(
                            CodeGenError::UndefinedVariable(name.clone(), span.clone()),
                        )?;
                        // Type/Mutability checked by Type Checker already, but double check type?
                        let expected_llvm_type = var_info.ty.to_llvm_basic_type(self.context).unwrap();
                        if compiled_rhs_value.get_type() != expected_llvm_type {
                            return Err(CodeGenError::InvalidType(
                                format!(
                                    "Assignment type mismatch for variable '{}': expected {:?}, found {:?}",
                                    name, expected_llvm_type, compiled_rhs_value.get_type()
                                ),
                                span,
                            ));
                        }
                        // Generate store
                        self.builder.build_store(var_info.ptr, compiled_rhs_value);
                        Ok(compiled_rhs_value) // Return assigned value
                    }
                    // Case 2: Index Assignment: target is IndexAccess { target: vec_expr, index }
                    ExpressionKind::IndexAccess { target: vec_expr, index } => {
                        // Compile vector handle and index
                        let vec_handle_val = self.compile_expression(vec_expr)?;
                        let index_val = self.compile_expression(index)?;
                        let vec_handle_ptr = match vec_handle_val {
                            BasicValueEnum::PointerValue(pv) => pv,
                            _ => {
                                return Err(CodeGenError::InvalidType(
                                    format!(
                                        "Expected vector handle to be a pointer, found {:?}",
                                        vec_handle_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        };
                        let index_i64 = match index_val {
                            BasicValueEnum::IntValue(iv) => {
                                if iv.get_type().get_bit_width() == 64 {
                                    iv
                                } else {
                                    return Err(CodeGenError::InvalidType(
                                        format!(
                                            "Index must be i64, found {:?}",
                                            index_val.get_type()
                                        ),
                                        span,
                                    ))
                                }
                            }
                            _ => {
                                return Err(CodeGenError::InvalidType(
                                    format!(
                                        "Expected index to be an integer, found {:?}",
                                        index_val.get_type()
                                    ),
                                    span,
                                ))
                            }
                        };

                        // Get element type from vec_expr's resolved type
                        let target_resolved_type = vec_expr.resolved_type.borrow().clone();
                        let elem_toy_type = match target_resolved_type {
                            Some(Type::Vector(et)) => *et,
                            _ => return Err(CodeGenError::InvalidType(
                                format!("{} Cannot index into non-vector type {:?}", span, target_resolved_type),
                                span,
                            )),
                        };
                        let elem_llvm_type = elem_toy_type.to_llvm_basic_type(self.context).unwrap();

                        // Type check RHS value (redundant if TC works, but safer)
                        if compiled_rhs_value.get_type() != elem_llvm_type { /* Error */ }

                        // Get pointer to element slot using vec_get_ptr
                        let vec_get_ptr_func = self.get_or_declare_vec_get_ptr();
                        let elem_ptr_i8 = match self.builder.build_call(vec_get_ptr_func, &[vec_handle_ptr.into(), index_i64.into()], "elem_ptr_i8") {
                            Ok(el) => el,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during vector get pointer: {}", e),
                                    span,
                                ))
                            }
                        }
                            .try_as_basic_value().left().ok_or(
                                CodeGenError::LlvmError(
                                    "Failed to get element pointer from vector".to_string(),
                                    span.clone(),
                                )
                        )?.into_pointer_value();
                        // TODO: Check for null ptr return from vec_get_ptr

                        // Cast i8* to typed pointer (e.g., i64*, float*)
                        let elem_ptr_typed = match self.builder.build_pointer_cast(
                            elem_ptr_i8,
                            self.context.ptr_type(AddressSpace::default()),
                            "elem_ptr_typed"
                        ){
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(
                                    format!("LLVM error during pointer cast: {}", e),
                                    span,
                                ))
                            }
                        }; // Cast to i8* first

                        // Generate store to write RHS value into the element slot
                        self.builder.build_store(elem_ptr_typed, compiled_rhs_value);

                        // Assignment expression returns the assigned value
                        Ok(compiled_rhs_value)
                    }
                    // Invalid L-value (Parser should prevent, but error defensively)
                    _ => Err(CodeGenError::InvalidType(
                        format!(
                            "Invalid L-value for assignment: expected variable or index access, found {:?}",
                            target.kind
                        ),
                        span,
                    )),
                } // End match target.kind
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
                                "If condition must be boolean (i1), found {:?}",
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
                            "If branches must have the same type: then {:?}, else {:?}",
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
                // 1. Infer element type (Type Checker should have ensured consistency)
                // We need the actual type here for size calculation. Re-check first element? Risky.
                // Assume type checker worked and passed info somehow, or re-infer.
                if elements.is_empty() {
                    // How to handle empty vector literal? Need type context.
                    return Err(CodeGenError::InvalidAstNode(
                        "Cannot codegen empty vector literal without type context".to_string(),
                        span,
                    ));
                }
                // Infer from first element - relies on type checker having run correctly
                let first_elem_val = self.compile_expression(&elements[0])?;
                let elem_llvm_type = first_elem_val.get_type();
                let elem_toy_type =
                    match elem_llvm_type {
                        // Basic reverse map
                        BasicTypeEnum::FloatType(_) => Type::Float,
                        BasicTypeEnum::IntType(it) if it.get_bit_width() == 1 => Type::Bool,
                        BasicTypeEnum::IntType(_) => Type::Int,
                        BasicTypeEnum::PointerType(_) => Type::String, // Assuming string ptr
                        _ => return Err(CodeGenError::InvalidType(
                            "Unsupported vector element type. todo: make ast annotate the types"
                                .to_string(),
                            span,
                        )),
                    };
                let elem_size = self.get_sizeof(&elem_toy_type).unwrap_or(0);
                if elem_size == 0 {
                    return Err(CodeGenError::InvalidType(
                        "Vector elements cannot be void".to_string(),
                        span,
                    ));
                }

                // 2. Call runtime vec_new(elem_size, capacity)
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
                .ok_or_else(|| {
                    CodeGenError::LlvmError("vec_new call failed".to_string(), span.clone())
                })?
                .into_pointer_value(); // Should return i8* handle

                // 3. Compile element expressions and store them
                let vec_get_ptr_func = self.get_or_declare_vec_get_ptr();
                for (i, elem_expr) in elements.iter().enumerate() {
                    let elem_val = self.compile_expression(elem_expr)?;
                    // Check type match (should be guaranteed by type checker)
                    if elem_val.get_type() != elem_llvm_type { /* Internal Compiler Error */ }

                    // Get pointer to element slot i: ptr = vec_get_ptr(handle, i)
                    let index_val = self.context.i64_type().const_int(i as u64, false);
                    let elem_ptr_i8 = match self.builder.build_call(
                        vec_get_ptr_func,
                        &[vec_handle.into(), index_val.into()],
                        "elem_ptr_i8",
                    ) {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during vec_get_ptr call: {}", e),
                                span.clone(),
                            ))
                        }
                    }
                    .try_as_basic_value()
                    .left()
                    .ok_or(CodeGenError::LlvmError(
                        "vec_get_ptr call failed".to_string(),
                        span.clone(),
                    ))?
                    .into_pointer_value();

                    // Cast i8* element slot pointer to actual element type pointer
                    let elem_ptr_typed = match self.builder.build_pointer_cast(
                        elem_ptr_i8,
                        self.context.ptr_type(AddressSpace::default()), // e.g., i64*, float*
                        "elem_ptr_typed",
                    ) {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during pointer cast: {}", e),
                                span.clone(),
                            ))
                        }
                    };

                    // Store the compiled element value into the slot
                    match self.builder.build_store(elem_ptr_typed, elem_val) {
                        Ok(_) => {}
                        Err(e) => {
                            return Err(CodeGenError::LlvmError(
                                format!("LLVM error during store: {}", e),
                                span.clone(),
                            ))
                        }
                    }
                }

                // 4. Vector literal expression evaluates to the handle (i8*)
                Ok(vec_handle.into())
            }

            // --- Index Access ---
            ExpressionKind::IndexAccess { target, index } => {
                // 1. Compile target (expect i8* vector handle) and index (expect i64)
                let target_handle = self.compile_expression(target)?;
                let index_val = self.compile_expression(index)?;

                // Type check results (basic)
                let vec_handle_ptr = match target_handle {
                    BasicValueEnum::PointerValue(pv) => pv, // Assume it's the i8* handle
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            "Target of index must be a vector".to_string(),
                            span,
                        ))
                    }
                };
                let index_i64 = match index_val {
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 64 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            "Index must be an integer".to_string(),
                            span,
                        ))
                    }
                };

                // 2. Determine Element Type (Need this from type checker!)
                // We lost the element type info here. We need the type checker pass
                // to potentially annotate the AST IndexAccess node with the vector's element type.
                let elem_toy_type =
                    expr.resolved_type.borrow().clone().expect(
                        "IndexAccess expression should have resolved type from type checker",
                    );
                let elem_llvm_type = elem_toy_type.to_llvm_basic_type(self.context).unwrap();

                // 3. Call runtime vec_get_ptr(handle, index) -> i8*
                let vec_get_ptr_func = self.get_or_declare_vec_get_ptr();
                let elem_ptr_i8 = match self.builder.build_call(
                    vec_get_ptr_func,
                    &[vec_handle_ptr.into(), index_i64.into()],
                    "elem_ptr_i8",
                ) {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("LLVM error during vec_get_ptr call: {}", e),
                            span,
                        ))
                    }
                }
                .try_as_basic_value()
                .left()
                .ok_or(CodeGenError::LlvmError(
                    "vec_get_ptr call failed".to_string(),
                    span.clone(),
                ))?
                .into_pointer_value();

                // 4. Cast i8* element slot pointer to actual element type pointer
                let elem_ptr_typed = match self.builder.build_pointer_cast(
                    elem_ptr_i8,
                    elem_llvm_type.ptr_type(AddressSpace::default()),
                    "elem_ptr_typed",
                ) {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(
                            format!("LLVM error during pointer cast: {}", e),
                            span.clone(),
                        ))
                    }
                };

                // 5. Load the value from the element pointer
                let loaded_val = self
                    .builder
                    .build_load(elem_llvm_type, elem_ptr_typed, "load_idx")
                    .map_err(|e| CodeGenError::LlvmError(format!("Load failed: {}", e), span))?;

                Ok(loaded_val) // Return the loaded element value
            }
        } // End match expr
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
        // Return cached function if available
        if let Some(func) = self.print_str_ln_wrapper_func {
            return Ok(func);
        }

        // Get the print_str wrapper first
        let str_wrapper = self.get_or_declare_print_str_wrapper()?;

        // Create print_str_ln wrapper with no parameters (just prints a newline)
        let fn_type = self.context.void_type().fn_type(&[], false);
        let function = self
            .module
            .add_function("print_str_ln_wrapper", fn_type, None);
        let entry_bb = self.context.append_basic_block(function, "entry");

        // Save current position
        let current_block = self.builder.get_insert_block();

        // Position at the wrapper function's entry block
        self.builder.position_at_end(entry_bb);

        // Create global string once (inside the wrapper function)
        let global_val = self.create_global_string("g_newline_constant", "\n", Linkage::Private);

        // Get pointer to the global string
        let zero_idx = self.context.i32_type().const_int(0, false);
        let string_type = global_val.get_value_type().into_array_type();
        let ptr_val = unsafe {
            self.builder.build_in_bounds_gep(
                string_type,
                global_val.as_pointer_value(),
                &[zero_idx, zero_idx],
                "newline_ptr",
            )
        }
        .map_err(|e| {
            CodeGenError::LlvmError(
                format!("LLVM error during newline pointer generation: {}", e),
                Span::default(),
            )
        })?;

        // Call print_str with the newline
        self.builder.build_call(str_wrapper, &[ptr_val.into()], "");

        // Return from wrapper
        self.builder.build_return(None);

        // Restore original position
        if let Some(block) = current_block {
            self.builder.position_at_end(block);
        }

        // Cache the function
        self.print_str_ln_wrapper_func = Some(function);

        Ok(function)
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

    // Get size of a ToyLang Type in bytes (basic implementation)
    fn get_sizeof(&self, ty: &Type) -> Option<u64> {
        // This needs target machine data layout ideally! Hardcoding for now. Assumes 64-bit.
        match ty {
            Type::Int | Type::Float => Some(8), // i64, f64
            Type::Bool => Some(1),              // i1 / bool
            Type::String => Some(8),            // i8* pointer size
            Type::Vector(_) => Some(8),         // Handle (pointer) size
            Type::Void => Some(0),              // Or None? Let's use 0.
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
                // Compile the value expression
                let compiled_value = self.compile_expression(value)?;
                self.compile_var_let_stmt(name, type_ann, compiled_value, false)
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

            StatementKind::VarBinding {
                name,
                type_ann,
                value,
            } => {
                // Compile the value expression
                let compiled_value = self.compile_expression(value)?;

                // Let statement yields no value itself
                self.compile_var_let_stmt(name, type_ann, compiled_value, true)
            }

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
                                "While condition must be boolean (i1), found {:?}",
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
                    let _ = self.compile_expression(init_expr)?;
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
                                        "For loop condition must be boolean (i1), found {:?}",
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
                                "If condition must be boolean (i1), found {:?}",
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
        type_ann: &Option<TypeNode>,
        compiled_value: BasicValueEnum,
        is_mutable: bool,
    ) -> CompileStmtResult<'ctx> {
        // convert type_ann: TypeNode to Type with type_node_to_type
        let type_ann = match type_ann {
            Some(type_node) => Some(type_node_to_type(type_node)),
            None => None, // Default to Float if no type annotation
        };
        // Determine the type: from annotation or inferred (basic inference for now)
        let var_type = match type_ann {
            Some(ann_ty) => {
                // TODO: Check if compiled_value type matches ann_ty (or is convertible)
                ann_ty.clone()
            }
            None => {
                // Basic inference from value (replace with proper type checker)
                match compiled_value {
                    BasicValueEnum::IntValue(iv) => {
                        if iv.get_type().get_bit_width() == 1 {
                            Type::Bool
                        } else {
                            Type::Int
                        }
                    }
                    BasicValueEnum::FloatValue(_) => Type::Float,
                    _ => {
                        return Err(CodeGenError::InvalidType(
                            format!(
                                "Cannot infer type for let binding [internal] [{}:{}]",
                                file!(),
                                line!()
                            ),
                            Span::default(),
                        ))
                    }
                }
            }
        };

        // Allocate based on determined type
        let alloca = self.create_entry_block_alloca(name, var_type.clone());
        self.builder.build_store(alloca, compiled_value);
        // Store type info along with pointer
        self.variables.insert(
            name.clone(),
            VariableInfo {
                ptr: alloca,
                ty: var_type,
                is_mutable,
            },
        );
        Ok((None, false)) // Let statement yields no value itself
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

// src/codegen.rs

use crate::ast::{
    BinaryOperator, ComparisonOperator, Expression, Program, Statement, UnaryOperator,
};
// Added BasicValueEnum
use crate::types::Type;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::passes::{PassManager, PassManagerBuilder};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::values::AnyValue;
use inkwell::values::{BasicMetadataValueEnum, BasicValue};
use inkwell::values::{BasicValueEnum, FunctionValue, GlobalValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::path::Path;

// --- CodeGenError --- (UndefinedVariable is still relevant)
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenError {
    InvalidAstNode(String),
    LlvmError(String),
    // UnknownOperator(BinaryOperator), // Covered by expression compilation
    UndefinedVariable(String),
    UndefinedFunction(String),    // Added
    FunctionRedefinition(String), // Added
    IncorrectArgumentCount {
        // Added
        func_name: String,
        expected: usize,
        found: usize,
    },
    PrintArgError(String),
    InvalidType(String), // For type mismatches or unsupported operations
    InvalidUnaryOperation(String), // For later
    InvalidBinaryOperation(String), // For type mismatches
    MissingTypeInfo(String), // If type info is needed but missing
    AssignmentToImmutable(String),
}
impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenError::InvalidAstNode(msg) => write!(f, "Invalid AST Node: {}", msg),
            CodeGenError::LlvmError(msg) => write!(f, "LLVM Error: {}", msg),
            CodeGenError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            CodeGenError::UndefinedFunction(name) => {
                write!(f, "Codegen Error: Undefined function '{}'", name)
            }
            CodeGenError::FunctionRedefinition(name) => {
                write!(f, "Codegen Error: Function '{}' redefined", name)
            }
            CodeGenError::IncorrectArgumentCount {
                func_name,
                expected,
                found,
            } => write!(
                f,
                "Codegen Error: Call to function '{}' expected {} arguments, found {}",
                func_name, expected, found
            ),
            CodeGenError::PrintArgError(msg) => write!(f, "Built-in 'print' Error: {}", msg),
            CodeGenError::InvalidType(msg) => write!(f, "Codegen Type Error: {}", msg),
            CodeGenError::InvalidBinaryOperation(msg) => {
                write!(f, "Codegen Binary Op Error: {}", msg)
            }
            CodeGenError::MissingTypeInfo(name) => {
                write!(f, "Codegen Error: Missing type info for '{}'", name)
            }
            CodeGenError::InvalidUnaryOperation(msg) => {
                write!(f, "Codegen Unary Op Error: {}", msg)
            }
            CodeGenError::AssignmentToImmutable(msg) => {
                write!(
                    f,
                    "Codegen Error: Cannot assign to immutable variable: {}",
                    msg
                )
            }
        }
    }
}
type CompileResult<'ctx, T> = Result<T, CodeGenError>; // Generic result

// Result now often returns BasicValueEnum as expressions can yield int, float, or bool
type CompileExprResult<'ctx> = Result<BasicValueEnum<'ctx>, CodeGenError>;
// CompileStmtResult remains similar, might yield BasicValueEnum if expr stmt returns non-float
type CompileStmtResult<'ctx> = Result<Option<BasicValueEnum<'ctx>>, CodeGenError>;

// Helper struct to store variable info (pointer + type)
#[derive(Debug, Clone, Copy)] // Copy is efficient as PointerValue/Type are Copy
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
    functions: HashMap<String, (Vec<Type>, Type, FunctionValue<'ctx>)>,
    current_function_return_type: Option<Type>, // Track expected return type
    fpm: PassManager<FunctionValue<'ctx>>,
    // Keep print wrapper stuff
    current_function: Option<FunctionValue<'ctx>>,
    print_float_wrapper_func: Option<FunctionValue<'ctx>>,
    print_int_wrapper_func: Option<FunctionValue<'ctx>>,
    print_bool_wrapper_func: Option<FunctionValue<'ctx>>,
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
        }
    }

    // Helper to get the type of an expression - NEEDS TYPE CHECKING PHASE ideally
    // For now, we infer based on structure, which is LIMITED and UNSAFE.
    fn infer_expression_type(&self, expr: &Expression) -> Result<Type, CodeGenError> {
        match expr {
            Expression::FloatLiteral(_) => Ok(Type::Float),
            Expression::IntLiteral(_) => Ok(Type::Int),
            Expression::BoolLiteral(_) => Ok(Type::Bool),
            Expression::Variable(name) => {
                self.variables
                    .get(name)
                    .map(|info| info.ty) // Get type from symbol table
                    .ok_or_else(|| CodeGenError::MissingTypeInfo(name.clone())) // Error if var used before defined (or type missing)
            }
            Expression::BinaryOp { op, left, right } => {
                // Very basic inference: assume result type matches operands
                // THIS IS WRONG: needs proper type checking based on operator rules!
                // E.g., int + int -> int, float + float -> float
                self.infer_expression_type(left)
            }
            Expression::ComparisonOp { .. } => {
                // Comparisons always return boolean
                Ok(Type::Bool)
            }
            Expression::FunctionCall { name, .. } => {
                // Look up function's declared return type
                self.functions
                    .get(name)
                    .map(|(_, ret_ty, _)| *ret_ty) // Get return type from map
                    .ok_or_else(|| CodeGenError::MissingTypeInfo(format!("function {}", name)))
            }
            _ => Err(CodeGenError::LlvmError("unexpected branch".to_string())),
        }
    }

    // Helper to create a global string constant (for format strings)
    fn create_global_string(&self, name: &str, value: &str, linkage: Linkage) -> GlobalValue<'ctx> {
        let c_string = CString::new(value).expect("CString::new failed");
        // Use `to_bytes()` (length *without* null) and let LLVM add the null terminator (`false`)
        // This ensures the type has the correct size (e.g., [4 x i8] for "%f\n")
        let string_val = self.context.const_string(c_string.to_bytes(), false);

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
        let llvm_type = ty.to_llvm_basic_type(self.context);
        // todo: check if unwrap is safe
        temp_builder.build_alloca(llvm_type, name).unwrap()
    }

    /// Compiles a single Expression node (used by statements)
    /// Returns a FloatValue representing the result of the expression.
    fn compile_expression(&mut self, expr: &Expression) -> CompileExprResult<'ctx> {
        match expr {
            Expression::FloatLiteral(value) => {
                Ok(self.context.f64_type().const_float(*value).into())
            }
            Expression::IntLiteral(value) => {
                // Assuming i64 for now
                Ok(self
                    .context
                    .i64_type()
                    .const_int(*value as u64, true)
                    .into()) // true=signed
            }
            Expression::BoolLiteral(value) => {
                Ok(self
                    .context
                    .bool_type()
                    .const_int(*value as u64, false)
                    .into()) // false=unsigned
            }

            Expression::Variable(name) => {
                match self.variables.get(name) {
                    Some(info) => {
                        let llvm_type = info.ty.to_llvm_basic_type(self.context);
                        let loaded_val = self
                            .builder
                            .build_load(llvm_type, info.ptr, name)
                            .map_err(|e| CodeGenError::LlvmError(format!("Load failed: {}", e)))?;
                        Ok(loaded_val) // Returns BasicValueEnum
                    }
                    None => Err(CodeGenError::UndefinedVariable(name.clone())),
                }
            }

            // Arithmetic Operations (Example: requires operands to be the same type for now)
            Expression::BinaryOp { op, left, right } => {
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
                                )))
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
                                )))
                            }
                        };
                        Ok(result.into())
                    }
                    _ => Err(CodeGenError::InvalidBinaryOperation(
                        format!("Type mismatch or unsupported types for operator {:?} (lhs: {:?}, rhs: {:?})", op, lhs.get_type(), rhs.get_type())
                    ))
                }
            }

            // Comparison Operations (Result is always Bool i1)
            Expression::ComparisonOp { op, left, right } => {
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
                                return Err(CodeGenError::InvalidBinaryOperation(format!(
                                    "Unsupported comparison {:?} for booleans",
                                    op
                                )))
                            }
                        }
                    }
                    _ => {
                        return Err(CodeGenError::InvalidBinaryOperation(format!(
                            "Type mismatch for comparison operator {:?} (lhs: {:?}, rhs: {:?})",
                            op,
                            lhs.get_type(),
                            rhs.get_type()
                        )))
                    }
                } {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(format!(
                            "LLVM error during comparison operation: {}",
                            e
                        )))
                    }
                };
                Ok(predicate.into()) // Comparison result is IntValue (i1)
            }
            // --- Unary Operation ---
            Expression::UnaryOp { op, operand } => {
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
                                        return Err(CodeGenError::LlvmError(format!(
                                            "LLVM error during integer negation: {}",
                                            e
                                        )))
                                    }
                                })
                                .into())
                            }
                            BasicValueEnum::FloatValue(fv) => {
                                Ok((match self.builder.build_float_neg(fv, "fneg_tmp") {
                                    Ok(val) => val,
                                    Err(e) => {
                                        return Err(CodeGenError::LlvmError(format!(
                                            "LLVM error during float negation: {}",
                                            e
                                        )))
                                    }
                                })
                                .into())
                            }
                            _ => Err(CodeGenError::InvalidUnaryOperation(format!(
                                "Cannot apply arithmetic negate '-' to type {:?}",
                                compiled_operand.get_type()
                            ))),
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
                                        return Err(CodeGenError::LlvmError(format!(
                                            "LLVM error during boolean negation: {}",
                                            e
                                        )))
                                    }
                                })
                                .into())
                            }
                            _ => Err(CodeGenError::InvalidUnaryOperation(format!(
                                "Cannot apply logical not '!' to type {:?}",
                                compiled_operand.get_type()
                            ))),
                        }
                    }
                } // End match op
            }
            // --- Handle Function Calls ---
            Expression::FunctionCall { name, args } => {
                // --- >> SPECIAL CASE: Built-in print function << ---
                if name == "print" {
                    if args.len() != 1 {
                        return Err(CodeGenError::IncorrectArgumentCount {
                            func_name: "print".to_string(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let arg_value = self.compile_expression(&args[0])?; // Result is BasicValueEnum

                    // --- CHOOSE WRAPPER/FORMAT based on type ---
                    match arg_value {
                        BasicValueEnum::IntValue(iv) => {
                            if iv.get_type().get_bit_width() == 1 {
                                // Boolean
                                let wrapper = self.get_or_declare_print_bool_wrapper()?; // Need new wrapper
                                self.builder.build_call(wrapper, &[iv.into()], "");
                            } else {
                                // Integer (i64)
                                let wrapper = self.get_or_declare_print_int_wrapper()?; // Need new wrapper
                                self.builder.build_call(wrapper, &[iv.into()], "");
                            }
                        }
                        BasicValueEnum::FloatValue(fv) => {
                            let wrapper = self.get_or_declare_print_float_wrapper()?; // Rename old one
                            self.builder.build_call(wrapper, &[fv.into()], "");
                        }
                        _ => {
                            return Err(CodeGenError::PrintArgError("Unsupported type".to_string()))
                        }
                    }
                    // Print returns 0.0 float for now
                    Ok(self.context.f64_type().const_float(0.0).into())
                }
                // --- User Function Call ---
                else {
                    let (param_types, return_type, func_val) = self
                        .functions
                        .get(name)
                        .ok_or_else(|| CodeGenError::UndefinedFunction(name.clone()))?
                        .clone();

                    if args.len() != param_types.len() {
                        return Err(CodeGenError::IncorrectArgumentCount {
                            func_name: name.clone(),
                            expected: param_types.len(),
                            found: args.len(),
                        });
                    }

                    let mut compiled_args: Vec<BasicMetadataValueEnum> =
                        Vec::with_capacity(args.len());
                    for (i, arg_expr) in args.iter().enumerate() {
                        let arg_val = self.compile_expression(arg_expr)?;
                        let expected_llvm_type = param_types[i].to_llvm_basic_type(self.context);
                        // --- Type Checking/Conversion (Basic) ---
                        if arg_val.get_type() != expected_llvm_type {
                            // Attempt basic conversion (e.g., int to float) if needed later
                            // Or error out
                            return Err(CodeGenError::InvalidType(format!(
                                "Argument {} type mismatch for function '{}': expected {:?}, found {:?}",
                                i, name, expected_llvm_type, arg_val.get_type()
                            )));
                        }
                        compiled_args.push(arg_val.into());
                    }

                    let call_site_val =
                        match self.builder.build_call(func_val, &compiled_args, "calltmp") {
                            Ok(val) => val,
                            Err(e) => {
                                return Err(CodeGenError::LlvmError(format!(
                                    "LLVM error during function call: {}",
                                    e
                                )))
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
                                return_type.to_llvm_basic_type(self.context);
                            match call_site_val.try_as_basic_value().left() {
                                Some(ret_val) if ret_val.get_type() == expected_llvm_ret_type => {
                                    Ok(ret_val)
                                }
                                _ => Err(CodeGenError::LlvmError(format!(
                                    "Call to '{}' did not return expected type {:?}",
                                    name, expected_llvm_ret_type
                                ))),
                            }
                        }
                    }
                }
            } // End FunctionCall
            // --- Assignment ---
            Expression::Assignment { target, value } => {
                // 1. Compile the RHS value
                let compiled_value = self.compile_expression(value)?;

                // 2. Look up the target variable
                let var_info = self
                    .variables
                    .get(target)
                    .ok_or_else(|| CodeGenError::UndefinedVariable(target.clone()))?;

                // 3. Check mutability
                if !var_info.is_mutable {
                    return Err(CodeGenError::AssignmentToImmutable(target.clone()));
                }

                // 4. Check Type Match (Basic - needs proper type checking/conversion)
                let expected_llvm_type = var_info.ty.to_llvm_basic_type(self.context);
                if compiled_value.get_type() != expected_llvm_type {
                    // TODO: Implement type conversion or error properly
                    return Err(CodeGenError::InvalidType(format!(
                        "Type mismatch in assignment to '{}': variable is {:?}, value is {:?}",
                        target,
                        expected_llvm_type,
                        compiled_value.get_type()
                    )));
                }

                // 5. Generate the store instruction
                self.builder.build_store(var_info.ptr, compiled_value);

                // 6. Assignment expression returns the assigned value
                Ok(compiled_value)
            }

            // --- If Expression (Branches are now Expressions) ---
            Expression::IfExpr {
                condition,
                then_branch,
                else_branch,
            } => {
                // Compile condition, check if bool (i1)
                let cond_val = self.compile_expression(condition)?;
                let bool_cond = match cond_val {
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(format!(
                            "If condition must be boolean (i1), found {:?}",
                            cond_val.get_type()
                        )))
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
                    return Err(CodeGenError::InvalidType(format!(
                        "If branches must have the same type: then {:?}, else {:?}",
                        then_val.get_type(),
                        else_val.get_type()
                    )));
                }
                let phi_type = then_val.get_type();
                let phi_node = match self.builder.build_phi(phi_type, "ifexpr_tmp") {
                    Ok(val) => val,
                    Err(e) => {
                        return Err(CodeGenError::LlvmError(format!(
                            "LLVM error creating PHI node: {}",
                            e
                        )))
                    }
                };
                phi_node.add_incoming(&[(&then_val, then_end_bb), (&else_val, else_end_bb)]);

                Ok(phi_node.as_basic_value())
            } // End IfExpr case

            // --- Block Expression ---
            Expression::Block {
                statements,
                final_expression,
            } => {
                // --- Scoping (Important!) ---
                // Need to handle variable scopes if blocks introduce them.
                // For now, assume blocks share scope with parent. Add proper scoping later.
                // let original_vars = self.variables.clone(); // Save parent scope if needed

                // 1. Compile the statements sequentially
                for stmt in statements {
                    // Compile statement, ignore optional value using `?` to propagate error
                    self.compile_statement(stmt)?;
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

                // --- Restore Scope (if implemented) ---
                // self.variables = original_vars;

                Ok(result)
            }
        } // End match expr
    }

    // Helper to compile a block (Program) and return the value of its last ExpressionStmt
    // Returns Ok(None) if block is empty or has no final ExpressionStmt
    fn compile_block(&mut self, block_program: &Program) -> CompileStmtResult<'ctx> {
        let mut last_val: Option<BasicValueEnum<'ctx>> = None;
        // --- Scoping ---
        // Need proper lexical scoping here. For now, variables leak out/in.
        // let original_vars = self.variables.clone();
        for stmt in &block_program.statements {
            let stmt_val = self.compile_statement(stmt)?;
            if stmt_val.is_some() {
                last_val = stmt_val;
            }
        }
        // self.variables = original_vars;
        Ok(last_val)
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

    /// Compiles a single Statement node.
    /// Returns Ok(Some(FloatValue)) if it's an ExpressionStmt, Ok(None) for LetBinding, Err on failure.
    /// Compiles a single Statement node. Now runs FPM on function definitions.
    // --- Compile Statement (handles types) ---
    fn compile_statement(&mut self, stmt: &Statement) -> CompileStmtResult<'ctx> {
        match stmt {
            Statement::LetBinding {
                name,
                type_ann,
                value,
            } => {
                // Compile the value expression
                let compiled_value = self.compile_expression(value)?;
                self.compile_var_let_stmt(name, type_ann, compiled_value, false)
            }

            Statement::VarBinding {
                name,
                type_ann,
                value,
            } => {
                // Compile the value expression
                let compiled_value = self.compile_expression(value)?;

                // Let statement yields no value itself
                self.compile_var_let_stmt(name, type_ann, compiled_value, true)
            }

            Statement::ExpressionStmt(expr) => {
                // Compile the expression, result might be any basic type
                let value = self.compile_expression(expr)?;
                Ok(Some(value))
            }

            // --- While Statement ---
            Statement::WhileStmt { condition, body } => {
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
                        return Err(CodeGenError::InvalidType(format!(
                            "While condition must be boolean (i1), found {:?}",
                            cond_val.get_type()
                        )))
                    }
                };
                // Build conditional branch based on condition value
                self.builder
                    .build_conditional_branch(bool_cond, loop_bb, after_bb);

                // 4. Compile Loop Body Block
                self.builder.position_at_end(loop_bb);
                let _ = self.compile_block(body)?; // Compile the body statements, ignore value
                                                   // After body, unconditionally branch back to condition check
                if loop_bb.get_terminator().is_none() {
                    // Only branch if block wasn't terminated (e.g. by future return/break)
                    self.builder.build_unconditional_branch(cond_bb);
                }

                // 5. Position builder at the block after the loop
                self.builder.position_at_end(after_bb);

                // While statement yields no value
                Ok(None)
            } // End WhileStmt

            // --- If Statement (optional else, no return value/PHI needed) ---
            Statement::IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                // Compile condition, check bool...
                let cond_val = self.compile_expression(condition)?;
                let bool_cond = match cond_val {
                    BasicValueEnum::IntValue(iv) if iv.get_type().get_bit_width() == 1 => iv,
                    _ => {
                        return Err(CodeGenError::InvalidType(format!(
                            "If condition must be boolean (i1), found {:?}",
                            cond_val.get_type()
                        )))
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
                let _ = self.compile_block(then_branch)?; // Compile the Program block
                if then_bb.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(merge_bb);
                }

                // Compile ELSE branch (Program block)
                if let Some(else_prog) = else_branch {
                    self.builder.position_at_end(else_bb);
                    let _ = self.compile_block(else_prog)?; // Compile the Program block
                    if else_bb.get_terminator().is_none() {
                        self.builder.build_unconditional_branch(merge_bb);
                    }
                }

                self.builder.position_at_end(merge_bb);
                Ok(None)
            }

            Statement::FunctionDef {
                name,
                params,
                return_type_ann,
                body,
            } => {
                if self.functions.contains_key(name) { // Function already defined
                    return Err(CodeGenError::FunctionRedefinition(name.clone()));
                }

                // --- Determine Param and Return Types ---
                // Use annotations, default to float for now if missing (NEEDS TYPE CHECKING)
                // --- Determine Param and Return Types ---
                let toy_param_types: Vec<Type> = params
                    .iter()
                    .map(|(_, opt_ty)| opt_ty.unwrap_or(Type::Float)) // Default Float (improve later)
                    .collect();
                let toy_return_type = return_type_ann.unwrap_or(Type::Float); // Default Float (improve later)

                let llvm_param_types: Vec<BasicMetadataTypeEnum> = toy_param_types
                    .iter()
                    .map(|ty| ty.to_llvm_basic_type(self.context).into())
                    .collect();
                let fn_type = match toy_return_type {
                    Type::Float => self.context.f64_type().fn_type(&llvm_param_types, false),
                    Type::Int => self.context.i64_type().fn_type(&llvm_param_types, false),
                    Type::Bool => self.context.bool_type().fn_type(&llvm_param_types, false),
                    Type::Void => self.context.void_type().fn_type(&llvm_param_types, false),
                    // Add other types like Pointer, Array, Struct later if needed
                };

                let function = self.module.add_function(name, fn_type, None);
                // Store signature with FunctionValue
                self.functions.insert(
                    name.clone(),
                    (toy_param_types.clone(), toy_return_type, function),
                );

                // --- Setup Function Body Context ---
                let entry_block = self.context.append_basic_block(function, "entry");
                let original_builder_pos = self.builder.get_insert_block();
                let original_func = self.current_function;
                let original_vars = self.variables.clone();
                let original_ret_type = self.current_function_return_type; // Save outer return type

                self.builder.position_at_end(entry_block);
                self.current_function = Some(function);
                self.current_function_return_type = Some(toy_return_type); // Set expected return type
                self.variables.clear();

                // --- Allocate and Store Parameters (using determined types) ---
                for (i, (param_name, _)) in params.iter().enumerate() {
                    let param_toy_type = toy_param_types[i]; // Get the type we determined
                    let llvm_param = function.get_nth_param(i as u32).unwrap();
                    llvm_param.set_name(param_name);
                    // llvm_param should already have the correct LLVM type from fn_type

                    let param_alloca = self.create_entry_block_alloca(param_name, param_toy_type);
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

                // --- Compile Function Body ---
                let mut last_body_val: Option<BasicValueEnum<'ctx>> = None;
                let mut body_compile_err: Option<CodeGenError> = None;
                for body_stmt in &body.statements { 
                    match self.compile_statement(body_stmt) {
                        Ok(Some(val)) => last_body_val = Some(val),
                        Ok(None) => {} // Ignore None values (e.g., let bindings)
                        Err(e) => {
                            body_compile_err = Some(e);
                            break; // Stop on first error
                        }
                    }
                }   

                // --- Build Return (check type) ---
                if body_compile_err.is_none() {
                    match toy_return_type {
                        Type::Void => {
                            self.builder.build_return(None);
                        } // Return void
                        _ => {
                            // Expect Int, Float, Bool
                            if let Some(ret_val) = last_body_val {
                                // TODO: Check if ret_val type matches toy_return_type
                                // TODO: Add explicit conversions if needed/allowed
                                self.builder.build_return(Some(&ret_val));
                            } else {
                                // Implicit return - return default value for the type? Error?
                                let default_val: BasicValueEnum<'ctx> = match toy_return_type {
                                    Type::Int => self.context.i64_type().const_int(0, false).into(),
                                    Type::Bool => {
                                        self.context.bool_type().const_int(0, false).into()
                                    }
                                    _ => self.context.f64_type().const_float(0.0).into(), // Default float
                                };
                                self.builder.build_return(Some(&default_val));
                            }
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
                if body_compile_err.is_some() || !function.verify(true) {
                    // Handle function verification failure
                    if let Some(err) = body_compile_err {
                        return Err(err);
                    }
                    return Err(CodeGenError::LlvmError(
                        "Function verification failed".to_string(),
                    ));
                }
                self.fpm.run_on(&function);
                if !function.verify(true) { 
                    return Err(CodeGenError::LlvmError(
                        "Function verification failed after optimization".to_string(),
                    ));
                }

                Ok(None)
            } // End FunctionDef
        } // End match stmt
    }

    fn compile_var_let_stmt(
        &mut self,
        name: &String,
        type_ann: &Option<Type>,
        compiled_value: BasicValueEnum,
        is_mutable: bool,
    ) -> CompileStmtResult<'ctx> {
        // Determine the type: from annotation or inferred (basic inference for now)
        let var_type = match type_ann {
            Some(ann_ty) => {
                // TODO: Check if compiled_value type matches ann_ty (or is convertible)
                *ann_ty
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
                            "Cannot infer type for let binding".to_string(),
                        ))
                    }
                }
            }
        };

        // Allocate based on determined type
        let alloca = self.create_entry_block_alloca(name, var_type);
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
        Ok(None) // Let statement yields no value itself
    }
    // End compile_statement

    /// Compile the program into the module, generating a standard C main function
    /// that calls printf with the result of the last top-level expression.
    pub fn compile_program_to_module(&mut self, program: &Program) -> Result<(), CodeGenError> {
        // --- Reset state for this compilation unit ---
        self.functions.clear(); // Clear user function definitions (printf is handled separately)

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
                ));
            }
        } else {
            // Handle verification failure before optimization
            // unsafe { main_function.delete(); }
            return Err(CodeGenError::LlvmError(
                "Main function verification failed before optimization".to_string(),
            ));
        }

        Ok(()) // Module is ready
    }

    /// Emits the compiled module to an object file.
    /// Should be called *after* compile_program_to_module.
    pub fn emit_object_file(&self, output_path: &Path) -> Result<(), CodeGenError> {
        // --- Target Configuration ---
        // Determine Target Triple for M4 Pro Mac
        let target_triple = &TargetTriple::create("aarch64-apple-darwin"); // ARM64 macOS

        // Initialize required targets (native is usually sufficient for host compilation)
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| {
            CodeGenError::LlvmError(format!("Failed to initialize native target: {}", e))
        })?;
        // Alternatively, initialize specific targets: Target::initialize_aarch64(&InitializationConfig::default());

        // --- Target Lookup ---
        let target = Target::from_triple(target_triple).map_err(|e| {
            CodeGenError::LlvmError(format!(
                "Failed to get target for triple '{}': {}",
                target_triple, e
            ))
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
                CodeGenError::LlvmError(format!(
                    "Failed to create target machine for triple '{}'",
                    target_triple
                ))
            })?;

        // --- Emit Object File ---
        println!("Emitting object file to: {}", output_path.display());
        target_machine
            .write_to_file(
                &self.module,     // The LLVM module containing the compiled code
                FileType::Object, // Specify we want an object file
                output_path,
            )
            .map_err(|e| CodeGenError::LlvmError(format!("Failed to write object file: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;
    use std::path::Path;

    #[test]
    
    fn compile_float_literal_expression() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);


        let expr = Expression::FloatLiteral(5.43);
        let result = compiler.compile_expression(&expr);

        assert!(result.is_ok());
        assert_eq!(
            result
                .unwrap()
                .into_float_value()
                .print_to_string()
                .to_string(),
            "double 5.430000e+00"
        );
    }

    #[test]
    fn compile_undefined_variable_error() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        let expr = Expression::Variable("x".to_string());
        let result = compiler.compile_expression(&expr);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            CodeGenError::UndefinedVariable("x".to_string())
        );
    }

    #[test]
    fn compile_binary_operation_with_type_mismatch() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        let expr = Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(Expression::IntLiteral(5)),
            right: Box::new(Expression::FloatLiteral(5.43)),
        };
        let result = compiler.compile_expression(&expr);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CodeGenError::InvalidBinaryOperation(_)
        ));
    }

    #[test]
    fn emit_object_file_success() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let compiler = Compiler::new(&context, &builder, &module);

        let output_path = Path::new("test_output.o");
        let result = compiler.emit_object_file(output_path);

        assert!(result.is_ok());
        assert!(output_path.exists());
        std::fs::remove_file(output_path).unwrap();
    }

    #[test]
    fn emit_object_file_invalid_path() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let compiler = Compiler::new(&context, &builder, &module);

        let output_path = Path::new("/invalid_path/test_output.o");
        let result = compiler.emit_object_file(output_path);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CodeGenError::LlvmError(_)));
    }
    
    #[test]
    fn compile_function_definition() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);
    
        // Create a simple function: function add(a: Int, b: Int) -> Int { a + b }
        let params = vec![
            ("a".to_string(), Some(Type::Int)),
            ("b".to_string(), Some(Type::Int)),
        ];
        
        let body = Program {
            statements: vec![
                Statement::ExpressionStmt(Expression::BinaryOp {
                    op: BinaryOperator::Add,
                    left: Box::new(Expression::Variable("a".to_string())),
                    right: Box::new(Expression::Variable("b".to_string())),
                }),
            ],
        };
    
        let func_def = Statement::FunctionDef {
            name: "add".to_string(),
            params,
            return_type_ann: Some(Type::Int),
            body: Box::new(body),
        };
    
        // Compile the function definition
        let result = compiler.compile_statement(&func_def);
        
        // Verify compilation succeeded
        assert!(result.is_ok());
        
        // Check that the function was added to the compiler's function map
        assert!(compiler.functions.contains_key("add"));
        
        // Verify the function signature is correct
        let (param_types, return_type, _) = compiler.functions.get("add").unwrap();
        assert_eq!(param_types.len(), 2);
        assert_eq!(param_types[0], Type::Int);
        assert_eq!(param_types[1], Type::Int);
        assert_eq!(*return_type, Type::Int);
    }
}

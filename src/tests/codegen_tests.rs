#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use crate::codegen::*;
    use crate::ast::{def, defs, BinaryOperator, Expression, ExpressionKind, Program, StatementKind, TypeNode, TypeNodeKind};
    use inkwell::context::Context;
    use std::path::Path;
    use inkwell::values::AnyValue;
    use crate::typechecker::TypeChecker;
    use crate::types::Type;

    #[test]

    fn compile_float_literal_expression() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        let expr = ExpressionKind::FloatLiteral(5.43);
        let result = compiler.compile_expression(&def(expr));

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

        let expr = Expression {
            kind: ExpressionKind::Variable("x".to_string()),
            span: Default::default(),
            resolved_type: RefCell::new(None),
        };
        let result = compiler.compile_expression(&expr);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            CodeGenError::UndefinedVariable("x".to_string(), Default::default())
        );
    }

    #[test]
    fn compile_binary_operation_with_type_mismatch() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        let expr = def(ExpressionKind::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(def(ExpressionKind::IntLiteral(5))),
            right: Box::new(def(ExpressionKind::FloatLiteral(5.43))),
        });

        let result = compiler.compile_expression(&expr);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CodeGenError::InvalidBinaryOperation(_, ..)
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
        assert!(matches!(
            result.unwrap_err(),
            CodeGenError::LlvmError(_, ..)
        ));
    }

    #[test]
    fn compile_function_definition() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        // Create a simple function: function add(a: Int, b: Int) -> Int { a + b }
        let params = vec![
            ("a".to_string(), Some(TypeNode{
                kind: TypeNodeKind::Simple("int".to_string()),
                span: Default::default(),
            })),
            ("b".to_string(), Some(TypeNode{
                kind: TypeNodeKind::Simple("int".to_string()),
                span: Default::default(),
            })),
        ];

        let body = Program {
            statements: vec![
                defs(StatementKind::ExpressionStmt(def(
                    ExpressionKind::BinaryOp {
                        op: BinaryOperator::Add,
                        left: Box::new(def(ExpressionKind::Variable("a".to_string()))),
                        right: Box::new(def(ExpressionKind::Variable("b".to_string()))),
                    },
                ))),
                defs(StatementKind::ReturnStmt {
                    value: Some(def(*Box::new(ExpressionKind::Variable("a".to_string())))),
                }),
            ],
            span: Default::default(),
        };

        let func_def = StatementKind::FunctionDef {
            name: "add".to_string(),
            params,
            return_type_ann: Some(TypeNode {
                kind: TypeNodeKind::Simple("int".to_string()),
                span: Default::default(),
            }),
            body,
        };
        let programm = Program {
            statements: vec![defs(func_def.clone())],
            span: Default::default(),
        };
        // run typechecker on the function definition
        let mut typechecker = TypeChecker::new();
        let typecheck_result = typechecker.check_program(&programm);
        assert!(typecheck_result.is_ok());

        // Compile the function definition
        let result = compiler.compile_program_to_module(&programm);
        // print all errors
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }

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

    #[test]
    fn compile_function_with_explicit_int_return() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);
        // Create a function: fun returns_int(): int { return 10; }
        let body = Program {
            statements: vec![defs(StatementKind::ReturnStmt {
                value: Some(def(*Box::new(ExpressionKind::IntLiteral(10)))),
            })],
            span: Default::default(),
        };

        let func_def = StatementKind::FunctionDef {
            name: "returns_int".to_string(),
            params: vec![],
            return_type_ann: Some(TypeNode {
                kind: TypeNodeKind::Simple("int".to_string()),
                span: Default::default(),
            }),
            body,
        };

        // Compile the function definition
        let result = compiler.compile_statement(&defs(func_def));
        assert!(result.is_ok());

        // Verify the function signature has the correct return type
        let (_, return_type, _) = compiler.functions.get("returns_int").unwrap();
        assert_eq!(*return_type, Type::Int);
    }

    #[test]
    fn compile_function_with_explicit_float_return() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        // Create a function: fun returns_float(): float { return 3.14; }
        let body = Program {
            statements: vec![defs(StatementKind::ReturnStmt {
                value: Some(def(ExpressionKind::FloatLiteral(3.20))),
            })],
            span: Default::default(),
        };

        let func_def = defs(StatementKind::FunctionDef {
            name: "returns_float".to_string(),
            params: vec![],
            return_type_ann: Some(TypeNode {
                kind: TypeNodeKind::Simple("float".to_string()),
                span: Default::default(),
            }),
            body,
        });

        // Compile the function definition
        let result = compiler.compile_statement(&func_def);
        assert!(result.is_ok());

        // Verify the function signature has the correct return type
        let (_, return_type, _) = compiler.functions.get("returns_float").unwrap();
        assert_eq!(*return_type, Type::Float);
    }

    #[test]
    fn compile_function_with_explicit_bool_return() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        // Create a function: fun returns_bool(): bool { return true; }
        let body = Program {
            statements: vec![defs(StatementKind::ReturnStmt {
                value: Some(def(ExpressionKind::BoolLiteral(true))),
            })],
            span: Default::default(),
        };

        let func_def = defs(StatementKind::FunctionDef {
            name: "returns_bool".to_string(),
            params: vec![],
            return_type_ann: Some(TypeNode {
                kind: TypeNodeKind::Simple("bool".to_string()),
                span: Default::default(),
            }),
            body,
        });

        // Compile the function definition
        let result = compiler.compile_statement(&func_def);
        assert!(result.is_ok());

        // Verify the function signature has the correct return type
        let (_, return_type, _) = compiler.functions.get("returns_bool").unwrap();
        assert_eq!(*return_type, Type::Bool);
    }

    #[test]
    fn compile_function_with_explicit_void_return() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        // Create a function: fun returns_void(): void { return; }
        let body = Program {
            statements: vec![defs(StatementKind::ReturnStmt { value: None })],
            span: Default::default(),
        };

        let func_def = defs(StatementKind::FunctionDef {
            name: "returns_void".to_string(),
            params: vec![],
            return_type_ann: Some(TypeNode {
                kind: TypeNodeKind::Simple("void".to_string()),
                span: Default::default(),
            }),
            body,
        });

        // Compile the function definition
        let result = compiler.compile_statement(&func_def);
        assert!(result.is_ok());

        // Verify the function signature has the correct return type
        let (_, return_type, _) = compiler.functions.get("returns_void").unwrap();
        assert_eq!(*return_type, Type::Void);
    }

    #[test]
    fn compile_function_with_implicit_void_return() {
        let context = Context::create();
        let module = context.create_module("test");
        let builder = context.create_builder();
        let mut compiler = Compiler::new(&context, &builder, &module);

        // Create a function: fun implicit_void(): void { print_str("Implicit void"); }
        let body = Program {
            statements: vec![defs(StatementKind::ExpressionStmt(def(
                ExpressionKind::FunctionCall {
                    name: "print_str".to_string(),
                    args: vec![def(ExpressionKind::StringLiteral(
                        "Implicit void".to_string(),
                    ))],
                },
            )))],
            span: Default::default(),
        };

        let func_def = defs(StatementKind::FunctionDef {
            name: "implicit_void".to_string(),
            params: vec![],
            return_type_ann: Some(TypeNode {
                kind: TypeNodeKind::Simple("void".to_string()),
                span: Default::default(),
            }),
            body,
        });

        // Compile the function definition
        let result = compiler.compile_statement(&func_def);
        assert!(result.is_ok());

        // Verify the function signature has the correct return type
        let (_, return_type, _) = compiler.functions.get("implicit_void").unwrap();
        assert_eq!(*return_type, Type::Void);
    }
}
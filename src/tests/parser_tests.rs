#[cfg(test)]
mod tests {
    use crate::ast::{
        def, BinaryOperator, BinaryOperator::*, ComparisonOperator, Expression, ExpressionKind,
        ExpressionKind::*, LogicalOperator, StatementKind, StatementKind::*, TypeNodeKind,
        UnaryOperator,
    };
    use crate::lexer::Lexer;
    use crate::parser::{ParseError, Parser};
    use crate::token::TokenKind;

    // Helper builders
    fn num(val: f64) -> ExpressionKind {
        IntLiteral(val as i64)
    }
    fn var(name: &str) -> ExpressionKind {
        Variable(name.to_string())
    }
    fn let_stmt(name: &str, value: ExpressionKind) -> StatementKind {
        LetBinding {
            name: name.to_string(),
            type_ann: None,
            value: def(value),
        }
    }
    fn expr_stmt(expr: Expression) -> StatementKind {
        ExpressionStmt(expr)
    }

    #[test]
    fn test_parse_neg_number() {
        let input = "-42;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap(); // Expect Ok
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn test_parse_unary_negation() {
        let input = "let val = -10;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            LetBinding { value, .. } => match value.kind {
                UnaryOp {
                    ref op,
                    ref operand,
                } => {
                    assert_eq!(*op, UnaryOperator::Negate);
                    assert_eq!(operand.kind, IntLiteral(10));
                }
                _ => panic!("Expected UnaryOp"),
            },
            _ => panic!("Expected LetBinding"),
        }
    }

    #[test]
    fn test_parse_unary_not() {
        let input = "!true;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(Expression {
                kind: UnaryOp { op, operand },
                ..
            }) => {
                assert_eq!(*op, UnaryOperator::Not);
                assert_eq!(operand.kind, BoolLiteral(true));
            }
            _ => panic!("Expected ExpressionStmt(UnaryOp)"),
        }
    }

    #[test]
    fn test_parse_unary_precedence() {
        let input = "-5 + 10;"; // Should parse as (-5) + 10
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        // Expected: BinaryOp(Add, UnaryOp(Negate, 5), 10)
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(Expression {
                kind: BinaryOp { op, left, right },
                ..
            }) => {
                assert_eq!(*op, Add);
                match &left.kind {
                    UnaryOp {
                        op: UnaryOperator::Negate,
                        operand,
                    } => {
                        assert_eq!(operand.kind, IntLiteral(5));
                    }
                    _ => panic!("Expected UnaryOp(Negate)"),
                }
                assert_eq!(right.kind, num(10.0));
            }
            _ => panic!("Expected ExpressionStmt(BinaryOp)"),
        }
    }

    #[test]
    fn test_parse_let_statement() {
        let input = "let answer = 42;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap(); // Expect Ok

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            LetBinding {
                name,
                type_ann,
                value,
            } => {
                assert_eq!(name, "answer");
                assert_eq!(type_ann, &None);
                match &value.kind {
                    IntLiteral(val) => assert_eq!(*val, 42),
                    _ => panic!("Expected IntLiteral"),
                }
            }
            _ => panic!("Expected LetBinding"),
        }
        // Ensure parser consumed everything
        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_parse_expression_statement() {
        let input = "5 + 5;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                BinaryOp { op, left, right } => {
                    assert_eq!(*op, Add);
                    match &left.kind {
                        IntLiteral(val) => assert_eq!(*val, 5),
                        _ => panic!("Expected IntLiteral for left operand"),
                    }
                    match &right.kind {
                        IntLiteral(val) => assert_eq!(*val, 5),
                        _ => panic!("Expected IntLiteral for right operand"),
                    }
                }
                _ => panic!("Expected BinaryOp"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_parse_sequence() {
        let input = "let x = 10; let y = x; y + 5;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 3);

        // Check first statement: let x = 10;
        match &program.statements[0].kind {
            LetBinding {
                name,
                type_ann,
                value,
            } => {
                assert_eq!(name, "x");
                assert_eq!(type_ann, &None);
                match &value.kind {
                    IntLiteral(val) => assert_eq!(*val, 10),
                    _ => panic!("Expected IntLiteral for x"),
                }
            }
            _ => panic!("Expected LetBinding for first statement"),
        }

        // Check second statement: let y = x;
        match &program.statements[1].kind {
            LetBinding {
                name,
                type_ann,
                value,
            } => {
                assert_eq!(name, "y");
                assert_eq!(type_ann, &None);
                match &value.kind {
                    Variable(var_name) => assert_eq!(var_name, "x"),
                    _ => panic!("Expected Variable for y"),
                }
            }
            _ => panic!("Expected LetBinding for second statement"),
        }

        // Check third statement: y + 5;
        match &program.statements[2].kind {
            ExpressionStmt(expr) => match &expr.kind {
                BinaryOp { op, left, right } => {
                    assert_eq!(*op, Add);
                    match &left.kind {
                        Variable(name) => assert_eq!(name, "y"),
                        _ => panic!("Expected Variable for left operand"),
                    }
                    match &right.kind {
                        IntLiteral(val) => assert_eq!(*val, 5),
                        _ => panic!("Expected IntLiteral for right operand"),
                    }
                }
                _ => panic!("Expected BinaryOp"),
            },
            _ => panic!("Expected ExpressionStmt for third statement"),
        }

        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_parse_precedence_in_statement() {
        let input = "1 + 2 * 3;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);

        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                BinaryOp { op, left, right } => {
                    assert_eq!(*op, Add);

                    // Check left: 1
                    match &left.kind {
                        IntLiteral(val) => assert_eq!(*val, 1),
                        _ => panic!("Expected IntLiteral for left operand"),
                    }

                    // Check right: 2 * 3
                    match &right.kind {
                        BinaryOp {
                            op: inner_op,
                            left: inner_left,
                            right: inner_right,
                        } => {
                            assert_eq!(*inner_op, Multiply);

                            match &inner_left.kind {
                                IntLiteral(val) => assert_eq!(*val, 2),
                                _ => panic!("Expected IntLiteral for inner left operand"),
                            }

                            match &inner_right.kind {
                                IntLiteral(val) => assert_eq!(*val, 3),
                                _ => panic!("Expected IntLiteral for inner right operand"),
                            }
                        }
                        _ => panic!("Expected BinaryOp for right operand"),
                    }
                }
                _ => panic!("Expected BinaryOp"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }

        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_parse_error_missing_semicolon() {
        let input = "let x = 5"; // Missing semicolon
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_program();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);

        // Check error type but ignore the location
        match &errors[0] {
            ParseError::ExpectedToken { expected, .. } => {
                assert_eq!(*expected, TokenKind::Semicolon);
            }
            _ => panic!("Expected ExpectedToken error"),
        }

        // Parser should stop at EOF after recovery attempt fails
        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    fn bin_op(op: BinaryOperator, left: ExpressionKind, right: ExpressionKind) -> ExpressionKind {
        BinaryOp {
            op,
            left: Box::new(def(left)),
            right: Box::new(def(right)),
        }
    }

    // Check the error recovery test assertion again
    #[test]
    fn test_parse_error_recovery() {
        let input = "let a = 1; let b = ; a + b;"; // Invalid 'let b' statement
        println!("Input: {}", input);
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_program();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        errors.iter().for_each(|e| println!("Error: {:?}", e));

        // Should have detected one error
        assert_eq!(
            errors.len(),
            1,
            "Expected exactly one parse error for input: {}",
            input
        ); // Add message
           // The error is UnexpectedToken when parse_prefix expects an expression after '=' but finds ';'
        assert!(
            matches!(errors[0], ParseError::UnexpectedToken{ ref found, .. } if found == &TokenKind::Semicolon ),
            "Unexpected error type or token: {:?}",
            errors[0]
        );
    }

    // Let's add a test specifically for grouped expressions now
    #[test]
    fn test_grouped_expression_statement() {
        let input = "(1 + 2) * 3;";
        println!("Input: {}", input);
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = match parser.parse_program() {
            Ok(prog) => prog,
            Err(errors) => {
                eprintln!("Parsing Errors:");
                for e in errors {
                    eprintln!("- {}", e);
                }
                panic!("Failed to parse program");
            }
        };

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                BinaryOp { op, left, right } => {
                    assert_eq!(*op, Multiply);
                    match &left.kind {
                        BinaryOp {
                            op: inner_op,
                            left: inner_left,
                            right: inner_right,
                        } => {
                            assert_eq!(*inner_op, Add);
                            assert_eq!(inner_left.kind, num(1.0));
                            assert_eq!(inner_right.kind, num(2.0));
                        }
                        _ => panic!("Expected BinaryOp for left operand"),
                    }
                    assert_eq!(right.kind, num(3.0));
                }
                _ => panic!("Expected BinaryOp"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    // function tests

    #[test]
    fn test_parse_function_definition_no_params() {
        let input = "fun main() { 5; }";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            FunctionDef {
                name, params, body, ..
            } => {
                assert_eq!(name, "main");
                assert!(params.is_empty());
                assert_eq!(body.statements.len(), 1);
                match &body.statements[0].kind {
                    ExpressionStmt(expr) => assert_eq!(expr.kind, num(5.0)),
                    _ => panic!("Expected ExpressionStmt in function body"),
                }
            }
            _ => panic!("Expected FunctionDef statement"),
        }
    }

    #[test]
    fn test_parse_function_definition_with_params() {
        let input = "fun add(a, b) { let result = a + b; result; }";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            FunctionDef {
                name, params, body, ..
            } => {
                assert_eq!(name, "add");
                assert_eq!(
                    params,
                    &vec![("a".to_string(), None), ("b".to_string(), None)]
                );
                assert_eq!(body.statements.len(), 2);
                match &body.statements[0].kind {
                    LetBinding { name, value, .. } => {
                        assert_eq!(name, "result");
                        match &value.kind {
                            BinaryOp { op, left, right } => {
                                assert_eq!(*op, Add);
                                assert_eq!(left.kind, var("a"));
                                assert_eq!(right.kind, var("b"));
                            }
                            _ => panic!("Expected BinaryOp in function body"),
                        }
                    }
                    _ => panic!("Expected LetBinding in function body"),
                }
                match &body.statements[1].kind {
                    ExpressionStmt(expr) => assert_eq!(expr.kind, var("result")),
                    _ => panic!("Expected ExpressionStmt in function body"),
                }
            }
            _ => panic!("Expected FunctionDef statement"),
        }
    }

    #[test]
    fn test_parse_function_call_no_args() {
        let input = "my_func();";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                FunctionCall { name, args } => {
                    assert_eq!(name, "my_func");
                    assert!(args.is_empty());
                }
                _ => panic!("Expected FunctionCall expression"),
            },
            _ => panic!("Expected ExpressionStmt statement"),
        }
    }

    #[test]
    fn test_parse_function_call_with_args() {
        let input = "add(1, 2 * 3);";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                FunctionCall { name, args } => {
                    assert_eq!(name, "add");
                    assert_eq!(args.len(), 2);
                    assert_eq!(args[0].kind, num(1.0));
                    match &args[1].kind {
                        BinaryOp { op, left, right } => {
                            assert_eq!(*op, Multiply);
                            assert_eq!(left.kind, num(2.0));
                            assert_eq!(right.kind, num(3.0));
                        }
                        _ => panic!("Expected BinaryOp for second argument"),
                    }
                }
                _ => panic!("Expected FunctionCall expression"),
            },
            _ => panic!("Expected FunctionCall expression statement"),
        }
    }

    #[test]
    fn test_call_inside_expression() {
        let input = "1 + compute(x, 5);";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                BinaryOp { op, left, right } => {
                    assert_eq!(*op, Add);
                    assert_eq!(left.kind, num(1.0));
                    match &right.kind {
                        FunctionCall { name, args } => {
                            assert_eq!(name, "compute");
                            assert_eq!(args.len(), 2);
                            assert_eq!(args[0].kind, Variable("x".to_string()));
                            assert_eq!(args[1].kind, IntLiteral(5));
                        }
                        _ => panic!("Expected FunctionCall as right operand"),
                    }
                }
                _ => panic!("Expected BinaryOp expression"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }

    #[test]
    fn test_program_with_fun_and_call() {
        let input = r#"
             fun double(n) { n * 2; }
             let num = 10;
             double(num + 5);
         "#;
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 3);
        assert!(matches!(program.statements[0].kind, FunctionDef { .. }));

        assert!(matches!(program.statements[1].kind, LetBinding { .. }));

        match &program.statements[2].kind {
            ExpressionStmt(expr) => match &expr.kind {
                FunctionCall { name, .. } => assert_eq!(name, "double"),
                _ => panic!("Expected FunctionCall expression"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }

    #[test]
    fn test_parse_comparisons() {
        let input = "a > 5 == true;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        // Expected: (a > 5) == true
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                ComparisonOp { op, left, right } => {
                    assert_eq!(*op, ComparisonOperator::Equal);
                    // ... more detailed assertions ...
                }
                _ => panic!("Expected ComparisonOp expression"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }

    #[test]
    fn test_let_with_type() {
        let input = "let count: int = 100;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        match &program.statements[0].kind {
            LetBinding {
                name,
                type_ann,
                value,
            } => {
                assert_eq!(name, "count");
                match &type_ann.clone().unwrap().kind {
                    // Unwrap to get the TypeNode}
                    TypeNodeKind::Simple(t) => assert_eq!(t, "int"),
                    _ => panic!("Expected TypeNode"),
                }
                assert_eq!(value.kind, IntLiteral(100));
            }
            _ => panic!("Expected LetBinding"),
        }
    }

    #[test]
    fn test_parse_var_statement() {
        let input = "var counter: int = 0;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        match &program.statements[0].kind {
            VarBinding {
                name,
                type_ann,
                value,
            } => {
                assert_eq!(name, "counter");
                match &type_ann.clone().unwrap().kind {
                    TypeNodeKind::Simple(t) => assert_eq!(t, "int"),
                    _ => panic!("Expected TypeNode"),
                }
                assert_eq!(value.kind, IntLiteral(0));
            }
            _ => panic!("Expected VarBinding"),
        }
    }

    #[test]
    fn test_parse_index_assignment() {
        let input = "my_vec[0] = 100;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::ExpressionStmt(expr) => match &expr.kind {
                ExpressionKind::Assignment { target, value } => {
                    // Check target is IndexAccess
                    match &target.kind {
                        ExpressionKind::IndexAccess {
                            target: vec_expr,
                            index,
                        } => {
                            match &vec_expr.kind {
                                ExpressionKind::Variable(name) => assert_eq!(name, "my_vec"),
                                _ => panic!("Expected var target in index"),
                            }
                            match &index.kind {
                                ExpressionKind::IntLiteral(i) => assert_eq!(*i, 0),
                                _ => panic!("Expected int literal index"),
                            }
                        }
                        _ => panic!("Assignment target not IndexAccess"),
                    }
                    // Check value is IntLiteral(100)
                    assert_eq!(value.kind, ExpressionKind::IntLiteral(100));
                }
                _ => panic!("Expected Assignment Expression"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }

    #[test]
    fn test_parse_push_call() {
        // Parses like a normal function call
        let input = "push(my_vec, 5);";
        // todo
        // ... assert AST is ExpressionStmt(FunctionCall{ name:"push", args:[Variable("my_vec"), IntLit(5)] }) ...
    }

    #[test]
    fn test_parse_invalid_assign_target() {
        let input = "5 = x;";
        // ... assert error is InvalidAssignmentTarget ...
        let input2 = "(x+y) = 10;";
        // ... assert error is InvalidAssignmentTarget ...
    }

    #[test]
    fn test_parse_if_else_missing_else() {
        // Our parser currently doesnt require else for `if` expressions
        let input = "if (x) { 1; }";
        println!("Input: {}", input);
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_program();
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            IfStmt {
                condition,
                then_branch,
                else_branch,
            } => {
                assert!(else_branch.is_none());
                match &condition.kind {
                    Variable(name) => assert_eq!(name, "x"),
                    _ => panic!("Condition was not Variable"),
                }
                assert_eq!(then_branch.statements.len(), 1);
            }
            _ => panic!("Expected IfStmt statement"),
        }
    }

    #[test]
    fn test_parse_if_expression_in_let() {
        let input = "let result = if (x > 0) { 1 } else { 2 };";
        println!("Input: {}", input);
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            LetBinding {
                name,
                type_ann: _,
                value,
            } => {
                assert_eq!(name, "result");
                // Check value is an IfExpr
                match &value.kind {
                    IfExpr {
                        condition: _,
                        then_branch: _,
                        else_branch: _,
                    } => {
                        // Add detailed checks for condition/branches if needed
                    }
                    _ => panic!("Let value was not IfExpr"),
                }
            }
            _ => panic!("Expected LetBinding statement"),
        }
    }

    #[test]
    fn test_parse_if_expression_block_branches() {
        // Note: no semicolon after 1 or -1
        let input = "let result = if (x > 0) { print(1); 1 } else { print(0); 2 };";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            LetBinding { value, .. } => match &value.kind {
                IfExpr {
                    condition: _,
                    then_branch,
                    else_branch,
                } => {
                    // Check then branch is Block with print stmt and final expr 1
                    match &then_branch.kind {
                        Block {
                            statements,
                            final_expression,
                        } => {
                            assert_eq!(statements.len(), 1); // print(1);
                            assert!(final_expression.is_some());
                            match &final_expression {
                                Some(e) => assert_eq!(e.kind, IntLiteral(1)), // Check final expr
                                None => panic!("Expected final expression"),
                            }
                        }
                        _ => panic!("Then branch not a Block expression"),
                    }
                    // Check else branch is Block with print stmt and final expr 2
                    match &else_branch.kind {
                        Block {
                            statements,
                            final_expression,
                        } => {
                            assert_eq!(statements.len(), 1); // print(0);
                            assert!(final_expression.is_some());
                            match &final_expression {
                                Some(e) => assert_eq!(e.kind, IntLiteral(2)), // Check final expr
                                None => panic!("Expected final expression"),
                            }
                        }
                        _ => panic!("Else branch not a Block expression"),
                    }
                }
                _ => panic!("Let value was not IfExpr"),
            },
            _ => panic!("Expected LetBinding statement"),
        }
    }

    #[test]
    fn test_parse_if_expr_missing_else() {
        // If used as expression MUST have else
        let input = "let result = if (x > 0) { 1 };";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_program();
        assert!(result.is_err());
        // Error occurs in parse_if_expression when expecting 'else'
        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, ParseError::ExpectedToken { expected, .. } if *expected == TokenKind::Else)));
    }

    #[test]
    fn test_parse_while_statement() {
        let input = "while (count < 10) { print(count); count = count + 1; }";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            WhileStmt { condition, body } => {
                // Check condition is count < 10
                // ... assertions for condition ...
                // Check body has two statements (print, assignment)
                assert_eq!(body.statements.len(), 2);
                // ... assertions for body statements ...
            }
            _ => panic!("Expected WhileStmt"),
        }
        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_parse_for_statement_full() {
        let input = "for (i = 0; i < 10; i = i + 1) { print(i); }";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::ForStmt {
                initializer,
                condition,
                increment,
                body,
            } => {
                assert!(initializer.is_some());
                assert!(condition.is_some());
                assert!(increment.is_some());
                // ... add detailed checks for init (assign), cond (<), incr (assign) ...
                assert_eq!(body.statements.len(), 1); // print(i);
            }
            _ => panic!("Expected ForStmt"),
        }
        assert_eq!(parser.current_token.kind, TokenKind::Eof);
    }

    #[test]
    fn test_parse_for_statement_empty_parts() {
        let input = "for (;;) { print(1); }"; // Infinite loop
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::ForStmt {
                initializer,
                condition,
                increment,
                body,
            } => {
                assert!(initializer.is_none());
                assert!(condition.is_none());
                assert!(increment.is_none());
                assert_eq!(body.statements.len(), 1);
            }
            _ => panic!("Expected ForStmt"),
        }
    }

    #[test]
    fn test_parse_return_statement_with_value() {
        let input = "return x + 1;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap(); // Assume parsing single statement
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::ReturnStmt { value } => {
                assert!(value.is_some());
                // Check value is BinaryOp{ Add, Variable("x"), IntLit(1) }
            }
            _ => panic!("Expected ReturnStmt"),
        }
    }
    #[test]
    fn test_parse_return_statement_no_value() {
        let input = "return;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::ReturnStmt { value } => {
                assert!(value.is_none());
            }
            _ => panic!("Expected ReturnStmt"),
        }
    }

    #[test]
    fn test_parse_vector_literal() {
        /* ... test [1, 2, 3] ... */
        let input = "[1, 2, 3];";

        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                ExpressionKind::VectorLiteral { elements } => {
                    assert_eq!(elements.len(), 3);
                    assert_eq!(elements[0].kind, IntLiteral(1));
                    assert_eq!(elements[1].kind, IntLiteral(2));
                    assert_eq!(elements[2].kind, IntLiteral(3));
                }
                _ => panic!("Expected VectorLiteral"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }
    #[test]
    fn test_parse_index_access() {
        /* ... test my_vec[0] ... */
        let input = "my_vec[0];";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                ExpressionKind::IndexAccess { target, index } => {
                    assert_eq!(target.kind, Variable("my_vec".to_string()));
                    assert_eq!(index.kind, IntLiteral(0));
                }
                _ => panic!("Expected IndexAccess"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }
    #[test]
    fn test_parse_vec_type_annotation() {
        /* ... test let x: vec<int> = ... */
        let input = "let x: vec<int> = [1, 2, 3];";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::LetBinding {
                name,
                type_ann,
                value,
            } => {
                assert_eq!(name, "x");
                match &type_ann.clone().unwrap().kind {
                    TypeNodeKind::Vector(t) => {
                        assert_eq!(t.kind, TypeNodeKind::Simple("int".to_string()));
                    }
                    _ => panic!("Expected TypeNode"),
                }
                match &value.kind {
                    ExpressionKind::VectorLiteral { elements } => {
                        assert_eq!(elements.len(), 3);
                        assert_eq!(elements[0].kind, IntLiteral(1));
                        assert_eq!(elements[1].kind, IntLiteral(2));
                        assert_eq!(elements[2].kind, IntLiteral(3));
                    }
                    _ => panic!("Expected VectorLiteral"),
                }
            }
            _ => panic!("Expected LetBinding"),
        }
    }

    #[test]
    fn test_parse_plus_plus() {
        let input = "x++;";
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            StatementKind::ExpressionStmt(s) => {
                // let value = Expression::new(
                //     ExpressionKind::BinaryOp {
                //         op,
                //         left: Box::new(left.clone()),
                //         right: Box::new(Expression {
                //             kind: ExpressionKind::IntLiteral(1),
                //             span: left.span.clone(),
                //             resolved_type: RefCell::new(None),
                //         }),
                //     },
                //     left.span.clone(),
                // );
                // left = Expression::new(
                //     ExpressionKind::Assignment {
                //         target: Box::new(left.clone()),
                //         value: Box::new(value),
                //     },
                //     left.span.clone(),
                // );
                match &s.kind {
                    ExpressionKind::Assignment { target, value } => match &target.kind {
                        ExpressionKind::Variable(name) => assert_eq!(name, "x"),
                        _ => panic!("Expected Variable for target"),
                    },
                    _ => panic!("Expected Assignment expression"),
                }
            }
            _ => panic!("Expected ExpressionStmt"),
        }
    }

    #[test]
    fn test_parse_logical_ops() {
        let input = "a && b || c;";
        // Expected: (a && b) || c
        let lexer = Lexer::new("test".to_string(), input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0].kind {
            ExpressionStmt(expr) => match &expr.kind {
                LogicalOp { op, left, right } => {
                    assert_eq!(*op, LogicalOperator::Or);
                    match &left.kind {
                        LogicalOp {
                            op: inner_op,
                            left: inner_left,
                            right: inner_right,
                        } => {
                            assert_eq!(*inner_op, LogicalOperator::And);
                            assert_eq!(inner_left.kind, Variable("a".to_string()));
                            assert_eq!(inner_right.kind, Variable("b".to_string()));
                        }
                        _ => panic!("Expected LogicalOp for left operand"),
                    }
                    assert_eq!(right.kind, Variable("c".to_string()));
                }
                _ => panic!("Expected LogicalOp expression"),
            },
            _ => panic!("Expected ExpressionStmt"),
        }
    }
}

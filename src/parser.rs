// src/parser.rs

// Use the new AST structures
use crate::ast::{BinaryOperator, ComparisonOperator, Expression, Program, Statement};
use crate::lexer::Lexer;
use crate::parser::Precedence::Lowest;
use crate::token::{keyword_to_type, Token};
use crate::types::Type;
use std::fmt;

// --- ParseError --- (Adjust as needed)
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken { expected: String, found: Token },
    EndOfInput,
    ExpectedExpression,
    ExpectedIdentifier,
    ExpectedToken(Token), // Good for '=' or ';'
    // LetStmtMissingSemicolon, // Example specific error
    InvalidTypeAnnotation(String),
    InvalidAssignmentTarget(String), // Added for errors like "5 = x;"
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken { expected, found } => {
                write!(f, "Parse Error: Expected {}, found {:?}", expected, found)
            }
            ParseError::EndOfInput => write!(f, "Parse Error: Unexpected end of input"),
            ParseError::ExpectedExpression => write!(f, "Parse Error: Expected an expression"),
            ParseError::ExpectedIdentifier => write!(f, "Parse Error: Expected an identifier"),
            ParseError::ExpectedToken(token) => {
                write!(f, "Parse Error: Expected token {:?}", token)
            }
            ParseError::InvalidTypeAnnotation(type_name) => {
                write!(f, "Parse Error: Invalid type annotation '{}'", type_name)
            }
            ParseError::InvalidAssignmentTarget(type_name) => {
                write!(f, "Parse Error: Invalid assignment target '{}'", type_name)
            }
        }
    }
}

// Change result type alias to return Program or specific statement/expression types
pub type ParseResult<T> = Result<T, ParseError>;

// --- Operator Precedence ---
// Define precedence levels for all operators
#[derive(PartialEq, PartialOrd)] // Allow comparing levels
enum Precedence {
    Lowest,
    Assign,      // =
    Equals,      // ==, !=
    LessGreater, // >, <, >=, <=
    Sum,         // +, -
    Product,     // *, /
    Prefix,      // -X or !X (Add later)
    Call,        // myFunction(X)
}

// Map tokens to precedence levels
fn token_precedence(token: &Token) -> Precedence {
    match token {
        Token::Assign => Precedence::Assign, // = has low precedence
        Token::Equal | Token::NotEqual => Precedence::Equals,
        Token::LessThan | Token::GreaterThan | Token::LessEqual | Token::GreaterEqual => {
            Precedence::LessGreater
        }
        Token::Plus | Token::Minus => Precedence::Sum,
        Token::Star | Token::Slash => Precedence::Product,
        Token::LParen => Precedence::Call, // For function calls
        _ => Lowest,
    }
}

// Helper to convert Token to BinaryOperator
fn token_to_binary_op(token: &Token) -> Option<BinaryOperator> {
    match token {
        Token::Plus => Some(BinaryOperator::Add),
        Token::Minus => Some(BinaryOperator::Subtract),
        Token::Star => Some(BinaryOperator::Multiply),
        Token::Slash => Some(BinaryOperator::Divide),
        _ => None,
    }
}

// Map tokens to AST operators (split binary and comparison)
fn token_to_comparison_op(token: &Token) -> Option<ComparisonOperator> {
    match token {
        Token::LessThan => Some(ComparisonOperator::LessThan),
        Token::GreaterThan => Some(ComparisonOperator::GreaterThan),
        Token::Equal => Some(ComparisonOperator::Equal),
        Token::NotEqual => Some(ComparisonOperator::NotEqual),
        Token::LessEqual => Some(ComparisonOperator::LessEqual),
        Token::GreaterEqual => Some(ComparisonOperator::GreaterEqual),
        _ => None,
    }
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current_token: Token,
    peek_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut parser = Parser {
            lexer,
            current_token: Token::Eof,
            peek_token: Token::Eof,
        };
        parser.next_token();
        parser.next_token();
        parser
    }

    fn next_token(&mut self) {
        self.current_token = self.peek_token.clone();
        self.peek_token = self.lexer.next_token();
    }

    // Helper to check current token and advance if it matches
    fn expect_and_consume(&mut self, expected: Token) -> ParseResult<()> {
        if self.current_token == expected {
            self.next_token();
            Ok(())
        } else {
            Err(ParseError::ExpectedToken(expected.clone())) // Clone expected for error
        }
    }

    // --- Main Parsing Logic: Parse the whole program ---

    pub fn parse_program(&mut self) -> Result<Program, Vec<ParseError>> {
        // Loop until EOF, call parse_statement, collect results/errors
        let mut statements = Vec::new();
        let mut errors = Vec::new();

        while self.current_token != Token::Eof {
            match self.parse_statement() {
                Ok(stmt) => statements.push(stmt),
                Err(err) => {
                    errors.push(err);
                    // Basic error recovery
                    while self.current_token != Token::Semicolon
                        && self.current_token != Token::RBrace
                        && self.current_token != Token::Eof
                    {
                        // Also recover on RBrace
                        self.next_token();
                    }
                    if self.current_token == Token::Semicolon || self.current_token == Token::RBrace
                    {
                        self.next_token();
                    }
                }
            }
        }
        // ... return Ok(Program) or Err(errors) ...
        if errors.is_empty() {
            Ok(Program { statements })
        } else {
            Err(errors)
        }
    }

    // --- Statement Parsing ---

    // --- Statement Parsing ---
    fn parse_statement(&mut self) -> ParseResult<Statement> {
        match self.current_token {
            Token::Let => self.parse_let_or_var_statement(false),
            Token::Var => self.parse_let_or_var_statement(true), // Pass mutable flag
            Token::Fun => self.parse_function_definition(),      // Added fun
            _ => self.parse_expression_statement(),
        }
    }

    // Parses: fun IDENT ( [PARAM_LIST]? ) [: TYPE]? { PROGRAM } ;?
    fn parse_function_definition(&mut self) -> ParseResult<Statement> {
        self.next_token(); // Consume 'fun'
        let name = match &self.current_token {
            Token::Identifier(n) => n.clone(),
            _ => return Err(ParseError::ExpectedIdentifier),
        };
        self.next_token(); // Consume name
        self.expect_and_consume(Token::LParen)?;
        let params = self.parse_parameter_list()?; // Now handles types

        // Optional return type annotation
        let return_type_ann: Option<Type> = if self.current_token == Token::Colon {
            self.next_token(); // Consume ':'
            match &self.current_token {
                Token::Identifier(type_name) => match keyword_to_type(type_name) {
                    Some(t) => {
                        self.next_token();
                        Some(t)
                    }
                    None => return Err(ParseError::InvalidTypeAnnotation(type_name.clone())),
                },
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "return type name".to_string(),
                        found: self.current_token.clone(),
                    })
                }
            }
        } else {
            None
        };

        self.expect_and_consume(Token::LBrace)?;
        let body = self.parse_block_statements()?;
        if self.current_token == Token::Semicolon {
            self.next_token();
        } // Optional semi

        Ok(Statement::FunctionDef {
            name,
            params,
            return_type_ann,
            body: Box::new(body),
        })
    }

    // Parses: IDENT [: TYPE]? [, IDENT [: TYPE]?]*
    fn parse_parameter_list(&mut self) -> ParseResult<Vec<(String, Option<Type>)>> {
        let mut params = Vec::new();
        if self.current_token == Token::RParen {
            self.next_token();
            return Ok(params);
        }

        // Parse first param: IDENT [: TYPE]?
        let (first_name, first_type) = self.parse_single_param()?;
        params.push((first_name, first_type));

        while self.current_token == Token::Comma {
            self.next_token(); // Consume ','
            let (name, type_ann) = self.parse_single_param()?;
            params.push((name, type_ann));
        }

        self.expect_and_consume(Token::RParen)?;
        Ok(params)
    }

    // Helper to parse a single "IDENT [: TYPE]?"
    fn parse_single_param(&mut self) -> ParseResult<(String, Option<Type>)> {
        let name = match &self.current_token {
            Token::Identifier(n) => n.clone(),
            _ => return Err(ParseError::ExpectedIdentifier),
        };
        self.next_token(); // Consume identifier

        let type_ann = if self.current_token == Token::Colon {
            self.next_token(); // Consume ':'
            match &self.current_token {
                Token::Identifier(type_name) => match keyword_to_type(type_name) {
                    Some(t) => {
                        self.next_token();
                        Some(t)
                    }
                    None => return Err(ParseError::InvalidTypeAnnotation(type_name.clone())),
                },
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "type name".to_string(),
                        found: self.current_token.clone(),
                    })
                }
            }
        } else {
            None
        };

        Ok((name, type_ann))
    }

    // Helper to parse statements within { ... }
    fn parse_block_statements(&mut self) -> ParseResult<Program> {
        let mut statements = Vec::new();
        // Note: Does not collect errors like top-level parse_program, stops on first error.
        // Could be enhanced later.

        // Loop until RBrace or EOF
        while self.current_token != Token::RBrace && self.current_token != Token::Eof {
            let stmt = self.parse_statement()?; // Parse one statement
            statements.push(stmt);
        }

        // Expect closing '}'
        self.expect_and_consume(Token::RBrace)?;

        Ok(Program { statements })
    }

    // Combined parser for let/var: let/var IDENT [: TYPE]? = EXPRESSION ;
    fn parse_let_or_var_statement(&mut self, is_mutable: bool) -> ParseResult<Statement> {
        self.next_token(); // Consume 'let' or 'var'

        let name = match &self.current_token {
            Token::Identifier(n) => n.clone(),
            _ => return Err(ParseError::ExpectedIdentifier),
        };
        self.next_token(); // Consume identifier

        // --- Optional Type Annotation ---
        let type_ann: Option<Type> = if self.current_token == Token::Colon {
            self.next_token(); // Consume ':'
            match &self.current_token {
                // Check if identifier is a known type name
                Token::Identifier(type_name) => {
                    match keyword_to_type(type_name) {
                        Some(t) => {
                            self.next_token(); // Consume type identifier
                            Some(t)
                        }
                        None => return Err(ParseError::InvalidTypeAnnotation(type_name.clone())),
                    }
                }
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "type name".to_string(),
                        found: self.current_token.clone(),
                    })
                }
            }
        } else {
            None // No type annotation
        };

        self.expect_and_consume(Token::Assign)?; // Expect and consume initial '='
        let value_expr = self.parse_expression(Lowest)?;
        self.expect_and_consume(Token::Semicolon)?;

        if is_mutable {
            Ok(Statement::VarBinding {
                name,
                type_ann,
                value: value_expr,
            })
        } else {
            Ok(Statement::LetBinding {
                name,
                type_ann,
                value: value_expr,
            })
        }
    }

    // Parses: EXPRESSION ;
    fn parse_expression_statement(&mut self) -> ParseResult<Statement> {
        let expr = self.parse_expression(Lowest)?; // Use enum precedence
        self.expect_and_consume(Token::Semicolon)?;
        Ok(Statement::ExpressionStmt(expr))
    }

    // --- Expression Parsing (Pratt Parser - mostly unchanged internally) ---

    // --- Expression Parsing (Updated Pratt Parser) ---
    fn parse_expression(&mut self, precedence: Precedence) -> ParseResult<Expression> {
        // Parse prefix expression
        let mut left = self.parse_prefix()?;

        // Loop while next token is infix and has higher precedence
        while precedence < token_precedence(&self.current_token) {
            // Check CURRENT token now
            // Check if current token is a binary/comparison operator OR function call start
            match self.current_token {
                // Arithmetic Binary Ops
                Token::Plus | Token::Minus | Token::Star | Token::Slash => {
                    let op = token_to_binary_op(&self.current_token).unwrap(); // Known to be Some
                    let current_precedence = token_precedence(&self.current_token);
                    self.next_token(); // Consume operator
                    let right = self.parse_expression(current_precedence)?;
                    left = Expression::BinaryOp {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                // Comparison Ops
                Token::LessThan
                | Token::GreaterThan
                | Token::Equal
                | Token::NotEqual
                | Token::LessEqual
                | Token::GreaterEqual => {
                    let op = token_to_comparison_op(&self.current_token).unwrap(); // Known to be Some
                    let current_precedence = token_precedence(&self.current_token);
                    self.next_token(); // Consume operator
                    let right = self.parse_expression(current_precedence)?;
                    left = Expression::ComparisonOp {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                // Function Call
                Token::LParen => {
                    // Need to check if 'left' was actually a Variable
                    let func_name = match left {
                        Expression::Variable(name) => name,
                        _ => {
                            return Err(ParseError::UnexpectedToken {
                                expected: "function name".to_string(),
                                found: self.current_token.clone(),
                            })
                        }
                    };
                    self.next_token(); // Consume '('
                    let args = self.parse_argument_list()?; // Returns vec of expressions
                    left = Expression::FunctionCall {
                        name: func_name,
                        args,
                    };
                }
                Token::Assign => {
                    // Ensure the left side is a valid assignment target (Identifier/Variable)
                    let target_name = match left {
                        Expression::Variable(name) => name,
                        // Add cases later for field access (obj.field = ...) or array index (arr[i] = ...)
                        _ => {
                            return Err(ParseError::InvalidAssignmentTarget(format!("{:?}", left)))
                        }
                    };

                    let current_precedence = token_precedence(&self.current_token);
                    self.next_token(); // Consume '='

                    // Assignment is right-associative, so parse RHS with slightly lower precedence
                    // (or equal precedence if definition strictly requires right-associativity handling)
                    // Let's use current_precedence for now, assuming basic handling.
                    // To be truly right-associative (x = y = 5 -> x = (y = 5)), the recursive
                    // call might need precedence `current_precedence - 1` or similar adjustment.
                    // Sticking with `current_precedence` makes it effectively non-associative here,
                    // requiring parentheses for chained assignment like `x = (y = 5);`.
                    let value = self.parse_expression(current_precedence)?;

                    left = Expression::Assignment {
                        target: target_name,
                        value: Box::new(value),
                    };
                }
                // Not an operator we handle infix at this precedence
                _ => break,
            }
        } // End loop
        Ok(left)
    }

    // Parses prefix elements: literals, identifiers, grouped expr, maybe unary later
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        println!("Parsing Prefix");
        let current_token_clone = self.current_token.clone();
        let result = match self.current_token.clone() {
            Token::FloatNum(value) => Ok(Expression::FloatLiteral(value)),
            Token::IntNum(value) => Ok(Expression::IntLiteral(value)),
            Token::BoolLiteral(value) => Ok(Expression::BoolLiteral(value)),
            Token::Identifier(name) => Ok(Expression::Variable(name)),
            Token::LParen => self.parse_grouped_expression(), // Delegate, consumes ()
            // Add unary ops later e.g. Token::Minus, Token::Bang
            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: self.current_token.clone(),
            }),
        };
        // Consume the token associated with the prefix element only if successful
        // and only if its not a () block, because the parse_group_expression function already consumes the RParen )!
        if result.is_ok() && !matches!(current_token_clone, Token::LParen) {
            self.next_token();
        }
        result
    }

    // Parses `( EXPR )` - consumes the initial `(` and final `)`
    fn parse_grouped_expression(&mut self) -> ParseResult<Expression> {
        println!("Parsing Grouped Expression");
        self.next_token();
        println!("current_token: {:?}", self.current_token);
        let expr = self.parse_expression(Lowest)?; // Parse inner expr
        self.expect_and_consume(Token::RParen)?; // Expect and consume ')'
        Ok(expr)
    }

    // Helper to parse EXPRESSION [, EXPRESSION]* within parentheses for function calls
    fn parse_argument_list(&mut self) -> ParseResult<Vec<Expression>> {
        let mut args = Vec::new();

        // Handle empty arg list: foo()
        if self.current_token == Token::RParen {
            self.next_token(); // Consume ')'
            return Ok(args);
        }

        // Parse first argument expression
        args.push(self.parse_expression(Lowest)?);

        // Parse subsequent arguments (comma followed by expression)
        while self.current_token == Token::Comma {
            self.next_token(); // Consume ','
            args.push(self.parse_expression(Lowest)?);
        }

        // Expect closing ')'
        self.expect_and_consume(Token::RParen)?;

        Ok(args)
    }

    // `parse_let_expression` is removed, replaced by `parse_let_statement`
}

// --- Update Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOperator::*, Expression::*, Statement::*};
    use crate::lexer::Lexer;

    // Helper builders
    fn num(val: f64) -> Expression {
        IntLiteral(val as i64)
    }
    fn var(name: &str) -> Expression {
        Variable(name.to_string())
    }
    fn let_stmt(name: &str, value: Expression) -> Statement {
        LetBinding {
            name: name.to_string(),
            type_ann: None,
            value,
        }
    }
    fn expr_stmt(expr: Expression) -> Statement {
        ExpressionStmt(expr)
    }

    #[test]
    fn test_parse_let_statement() {
        let input = "let answer = 42;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap(); // Expect Ok

        assert_eq!(program.statements.len(), 1);
        assert_eq!(program.statements[0], let_stmt("answer", num(42.0)));
        // Ensure parser consumed everything
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parse_expression_statement() {
        let input = "5 + 5;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        assert_eq!(
            program.statements[0],
            expr_stmt(bin_op(Add, num(5.0), num(5.0)))
        );
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parse_sequence() {
        let input = "let x = 10; let y = x; y + 5;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 3);
        assert_eq!(program.statements[0], let_stmt("x", num(10.0)));
        assert_eq!(program.statements[1], let_stmt("y", var("x")));
        assert_eq!(
            program.statements[2],
            expr_stmt(bin_op(Add, var("y"), num(5.0)))
        );
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parse_precedence_in_statement() {
        let input = "1 + 2 * 3;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        let expected_expr = bin_op(Add, num(1.0), bin_op(Multiply, num(2.0), num(3.0)));
        // Note: Need to adjust bin_op helper if taking Box directly
        // Let's fix bin_op helper
        fn bin_op(op: BinaryOperator, left: Expression, right: Expression) -> Expression {
            BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        let expected_expr = bin_op(Add, num(1.0), bin_op(Multiply, num(2.0), num(3.0)));

        assert_eq!(program.statements[0], expr_stmt(expected_expr));
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parse_error_missing_semicolon() {
        let input = "let x = 5"; // Missing semicolon
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_program();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        // The error occurs when parse_let_statement expects a semicolon after the expression '5'
        assert_eq!(errors[0], ParseError::ExpectedToken(Token::Semicolon));
        // Parser should stop at EOF after recovery attempt fails
        assert_eq!(parser.current_token, Token::Eof);
    }

    fn bin_op(op: BinaryOperator, left: Expression, right: Expression) -> Expression {
        BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    // Check the error recovery test assertion again
    #[test]
    fn test_parse_error_recovery() {
        let input = "let a = 1; let b = ; a + b;"; // Invalid 'let b' statement
        println!("Input: {}", input);
        let lexer = Lexer::new(input);
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
            matches!(errors[0], ParseError::UnexpectedToken{ ref found, .. } if found == &Token::Semicolon ),
            "Unexpected error type or token: {:?}",
            errors[0]
        );
    }

    // Let's add a test specifically for grouped expressions now
    #[test]
    fn test_grouped_expression_statement() {
        let input = "(1 + 2) * 3;";
        println!("Input: {}", input);
        let lexer = Lexer::new(input);
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
        let expected_expr = bin_op(Multiply, bin_op(Add, num(1.0), num(2.0)), num(3.0));
        assert_eq!(program.statements[0], expr_stmt(expected_expr));
        assert_eq!(parser.current_token, Token::Eof);
    }

    // function tests

    #[test]
    fn test_parse_function_definition_no_params() {
        let input = "fun main() { 5; }";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            FunctionDef {
                name, params, body, ..
            } => {
                assert_eq!(name, "main");
                assert!(params.is_empty());
                assert_eq!(body.statements.len(), 1);
                assert_eq!(body.statements[0], expr_stmt(num(5.0)));
            }
            _ => panic!("Expected FunctionDef statement"),
        }
    }

    #[test]
    fn test_parse_function_definition_with_params() {
        let input = "fun add(a, b) { let result = a + b; result; }";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            FunctionDef {
                name, params, body, ..
            } => {
                assert_eq!(name, "add");
                // assert_eq!(params, &["a".to_string(), "b".to_string()]);
                assert_eq!(body.statements.len(), 2);
                assert_eq!(
                    body.statements[0],
                    let_stmt("result", bin_op(Add, var("a"), var("b")))
                );
                assert_eq!(body.statements[1], expr_stmt(var("result")));
            }
            _ => panic!("Expected FunctionDef statement"),
        }
    }

    #[test]
    fn test_parse_function_call_no_args() {
        let input = "my_func();";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            ExpressionStmt(FunctionCall { name, args }) => {
                assert_eq!(name, "my_func");
                assert!(args.is_empty());
            }
            _ => panic!("Expected FunctionCall expression statement"),
        }
    }

    #[test]
    fn test_parse_function_call_with_args() {
        let input = "add(1, 2 * 3);";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            ExpressionStmt(FunctionCall { name, args }) => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], num(1.0));
                assert_eq!(args[1], bin_op(Multiply, num(2.0), num(3.0)));
            }
            _ => panic!("Expected FunctionCall expression statement"),
        }
    }

    #[test]
    fn test_call_inside_expression() {
        let input = "1 + compute(x, 5);";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 1);
        let expected_call = FunctionCall {
            name: "compute".to_string(),
            args: vec![var("x"), num(5.0)],
        };
        assert_eq!(
            program.statements[0],
            expr_stmt(bin_op(Add, num(1.0), expected_call))
        );
    }

    #[test]
    fn test_program_with_fun_and_call() {
        let input = r#"
             fun double(n) { n * 2; }
             let num = 10;
             double(num + 5);
         "#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.statements.len(), 3);
        assert!(matches!(
            program.statements[0],
            Statement::FunctionDef { .. }
        ));
        assert!(matches!(
            program.statements[1],
            Statement::LetBinding { .. }
        ));
        assert!(matches!(
            program.statements[2],
            Statement::ExpressionStmt(Expression::FunctionCall { .. })
        ));
    }

    #[test]
    fn test_parse_comparisons() {
        let input = "a > 5 == true;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        // Expected: (a > 5) == true
        let expected_expr = ComparisonOp {
            op: ComparisonOperator::Equal,
            left: Box::new(ComparisonOp {
                op: ComparisonOperator::GreaterThan,
                left: Box::new(Variable("a".to_string())),
                right: Box::new(IntLiteral(5)), // Assume 5 is int
            }),
            right: Box::new(BoolLiteral(true)),
        };
        assert_eq!(program.statements[0], ExpressionStmt(expected_expr));
    }

    #[test]
    fn test_let_with_type() {
        let input = "let count: int = 100;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        assert_eq!(
            program.statements[0],
            LetBinding {
                name: "count".to_string(),
                type_ann: Some(Type::Int),
                value: IntLiteral(100)
            }
        );
    }

    #[test]
    fn test_parse_var_statement() {
        let input = "var counter: int = 0;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        assert_eq!(
            program.statements[0],
            VarBinding {
                name: "counter".to_string(),
                type_ann: Some(Type::Int),
                value: IntLiteral(0)
            }
        );
    }

    #[test]
    fn test_parse_assignment_expression() {
        let input = "x = y + 1;"; // Assign result of y+1 to x
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();
        assert_eq!(program.statements.len(), 1);
        match &program.statements[0] {
            ExpressionStmt(Assignment { target, value }) => {
                assert_eq!(target, "x");
                // Check value is BinaryOp { Add, Variable("y"), IntLiteral(1) }
                match &**value {
                    // Deref the Box
                    BinaryOp { op, left, right } => {
                        assert_eq!(*op, Add);
                        assert_eq!(*left, Box::new(Variable("y".to_string())));
                        assert_eq!(*right, Box::new(IntLiteral(1)));
                    }
                    _ => panic!("Assignment value was not BinaryOp"),
                }
            }
            _ => panic!("Expected Assignment Expression Statement"),
        }
    }
}

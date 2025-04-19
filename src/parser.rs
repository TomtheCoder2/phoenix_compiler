// src/parser.rs

// Use the new AST structures
use crate::ast::{BinaryOperator, Expression, NumberType, Program, Statement};
use crate::lexer::Lexer;
use crate::token::Token;
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
        }
    }
}

// Change result type alias to return Program or specific statement/expression types
pub type ParseResult<T> = Result<T, ParseError>;

// Function to get the precedence (binding power) of binary operators
fn precedence(token: &Token) -> u8 {
    match token {
        Token::Plus | Token::Minus => 1,
        Token::Star | Token::Slash => 2,
        // Other operators like assignment, comparison would have different levels
        _ => 0, // Not an infix operator or lowest precedence
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
            Token::Let => self.parse_let_statement(),
            Token::Fun => self.parse_function_definition(), // Added fun
            _ => self.parse_expression_statement(),
        }
    }

    // Parses: fun IDENT ( [IDENT [, IDENT]*]? ) { PROGRAM } ;? <-- Note: semicolon is optional/discouraged after }
    fn parse_function_definition(&mut self) -> ParseResult<Statement> {
        self.next_token(); // Consume 'fun'

        // Expect function name (Identifier)
        let name = match &self.current_token {
            Token::Identifier(n) => n.clone(),
            _ => return Err(ParseError::ExpectedIdentifier),
        };
        self.next_token(); // Consume function name

        // Expect '(' for parameters
        self.expect_and_consume(Token::LParen)?;

        // Parse parameter list
        let params = self.parse_parameter_list()?; // Expects ')' inside

        // Expect '{' for body
        self.expect_and_consume(Token::LBrace)?;

        // Parse the body (which is a sequence of statements, i.e., a Program)
        // We need a way to parse statements until '}'
        let body = self.parse_block_statements()?; // Expects '}' inside

        // Optional semicolon after '}' - let's just consume if present
        if self.current_token == Token::Semicolon {
            self.next_token();
        }

        Ok(Statement::FunctionDef {
            name,
            params,
            body: Box::new(body),
        })
    }

    // Helper to parse IDENT [, IDENT]* within parentheses
    fn parse_parameter_list(&mut self) -> ParseResult<Vec<String>> {
        let mut params = Vec::new();

        // Handle empty parameter list: fun foo() { ... }
        if self.current_token == Token::RParen {
            self.next_token(); // Consume ')'
            return Ok(params);
        }

        // Expect first parameter identifier
        match &self.current_token {
            Token::Identifier(name) => params.push(name.clone()),
            _ => return Err(ParseError::ExpectedIdentifier),
        }
        self.next_token(); // Consume identifier

        // Expect subsequent parameters (comma followed by identifier)
        while self.current_token == Token::Comma {
            self.next_token(); // Consume ','
            match &self.current_token {
                Token::Identifier(name) => params.push(name.clone()),
                _ => return Err(ParseError::ExpectedIdentifier),
            }
            self.next_token(); // Consume identifier
        }

        // Expect closing ')'
        self.expect_and_consume(Token::RParen)?;

        Ok(params)
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

    // Parses: let IDENT = EXPRESSION ;
    fn parse_let_statement(&mut self) -> ParseResult<Statement> {
        self.next_token(); // Consume 'let'

        let name = match &self.current_token {
            Token::Identifier(n) => n.clone(),
            _ => return Err(ParseError::ExpectedIdentifier),
        };
        self.next_token(); // Consume identifier

        self.expect_and_consume(Token::Assign)?; // Consume '='

        // Parse the expression for the value
        let value_expr = self.parse_expression(0)?; // Parse value expr

        // self.next_token();
        // Expect semicolon to terminate the statement
        self.expect_and_consume(Token::Semicolon)?;

        Ok(Statement::LetBinding {
            name,
            value: value_expr,
        })
    }

    // Parses: EXPRESSION ;
    fn parse_expression_statement(&mut self) -> ParseResult<Statement> {
        let expr = self.parse_expression(0)?; // Parse the expression itself
        // self.next_token();
        // Expect semicolon to terminate the statement
        self.expect_and_consume(Token::Semicolon)?;

        Ok(Statement::ExpressionStmt(expr))
    }

    // --- Expression Parsing (Pratt Parser - mostly unchanged internally) ---

    // --- Expression Parsing ---
    fn parse_expression(&mut self, min_precedence: u8) -> ParseResult<Expression> {
        // Parse prefix first
        let mut left = self.parse_prefix()?; // Consumes its token(s) on success

        // Loop for infix operators OR function calls after a potential prefix
        loop {
            match self.current_token {
                // Infix Binary Operators (check precedence)
                ref tok
                    if precedence(tok) > min_precedence && token_to_binary_op(tok).is_some() =>
                {
                    let op = token_to_binary_op(&self.current_token).unwrap();
                    let current_precedence = precedence(&self.current_token);
                    self.next_token(); // Consume operator

                    let right = self.parse_expression(current_precedence)?;
                    left = Expression::BinaryOp {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                // Function Call: If current token is '(', assume 'left' was the function name
                // This requires 'left' to have been parsed as an identifier (Variable node).
                Token::LParen => {
                    // Ensure 'left' was a variable expression
                    let func_name = match left {
                        Expression::Variable(name) => name,
                        _ => {
                            return Err(ParseError::UnexpectedToken {
                                expected: "Function name before '('".to_string(),
                                found: Token::LParen, // Or report the node type of 'left'
                            });
                        }
                    };

                    self.next_token(); // Consume '('
                    let args = self.parse_argument_list()?; // Expects ')' inside
                    left = Expression::FunctionCall {
                        name: func_name,
                        args,
                    };
                }
                // No more operators or calls relevant at this precedence level
                _ => break,
            }
        } // End loop

        Ok(left)
    }

    // Parses prefix elements: literals, identifiers, grouped expressions
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        match self.current_token.clone() {
            Token::Number(value) => {
                self.next_token();
                Ok(Expression::NumberLiteral(value))
            }
            Token::Identifier(name) => {
                // This could be a variable OR the start of a function call.
                // We parse it as a Variable here. The '(' check in parse_expression's loop handles calls.
                self.next_token();
                Ok(Expression::Variable(name))
            }
            Token::LParen => {
                self.next_token(); // Consume '('
                self.parse_grouped_expression()
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "number, identifier, or '('".to_string(),
                found: self.current_token.clone(),
            }),
        }
    }

    fn parse_grouped_expression(&mut self) -> ParseResult<Expression> {
        // LParen was consumed by the caller (parse_prefix)
        let expr = self.parse_expression(0)?;
        // Current token should now be RParen

        // Expect and consume ')'
        self.expect_and_consume(Token::RParen)?; // Handles check and consumption

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
        args.push(self.parse_expression(0)?);

        // Parse subsequent arguments (comma followed by expression)
        while self.current_token == Token::Comma {
            self.next_token(); // Consume ','
            args.push(self.parse_expression(0)?);
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
    fn num(val: NumberType) -> Expression {
        NumberLiteral(val)
    }
    fn var(name: &str) -> Expression {
        Variable(name.to_string())
    }
    fn let_stmt(name: &str, value: Expression) -> Statement {
        LetBinding {
            name: name.to_string(),
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
            Statement::FunctionDef { name, params, body } => {
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
            Statement::FunctionDef { name, params, body } => {
                assert_eq!(name, "add");
                assert_eq!(params, &["a".to_string(), "b".to_string()]);
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
            Statement::ExpressionStmt(Expression::FunctionCall { name, args }) => {
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
            Statement::ExpressionStmt(Expression::FunctionCall { name, args }) => {
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
        let expected_call = Expression::FunctionCall {
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
}

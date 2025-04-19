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
        let mut statements = Vec::new();
        let mut errors = Vec::new();

        // Loop until End-of-File token
        while self.current_token != Token::Eof {
            match self.parse_statement() {
                Ok(stmt) => statements.push(stmt),
                Err(err) => {
                    errors.push(err);
                    // Error recovery: Skip tokens until the next semicolon or EOF
                    // to try and parse subsequent statements. This is basic recovery.
                    while self.current_token != Token::Semicolon && self.current_token != Token::Eof
                    {
                        self.next_token();
                    }
                    // Consume the semicolon that likely caused the error or ended the recovery attempt
                    if self.current_token == Token::Semicolon {
                        self.next_token();
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(Program { statements })
        } else {
            Err(errors) // Return all collected errors
        }
    }

    // --- Statement Parsing ---

    // Parses a single statement based on the current token
    fn parse_statement(&mut self) -> ParseResult<Statement> {
        match self.current_token {
            Token::Let => self.parse_let_statement(),
            // Anything else starts an expression statement
            _ => self.parse_expression_statement(),
        }
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

    // Renamed from parse_expression_recursive to clarify it parses expressions
    fn parse_expression(&mut self, min_precedence: u8) -> ParseResult<Expression> {
        // Parse prefix expression (number, identifier, '(', maybe unary later)
        // Note: we now get token ownership issues if parse_prefix fails and we try to consume
        // Let parse_prefix consume its token ON SUCCESS.
        let mut left = self.parse_prefix()?;


        // Loop for infix operators
        // Use peek_token for lookahead on operator precedence
        while min_precedence < precedence(&self.current_token) // Look ahead for operator
            && token_to_binary_op(&self.current_token).is_some()
        {
            // Now that peek_token is an operator, consume it (move peek to current)
            // self.next_token(); // Consume the operator, it's now current_token
            let op = token_to_binary_op(&self.current_token).unwrap();
            let current_precedence = precedence(&self.current_token);

            self.next_token(); // Consume the operator token, move to start of RHS

            let right = self.parse_expression(current_precedence)?; // Parse RHS

            left = Expression::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        } // Loop continues based on peek_token

        Ok(left)
    }

    // Parses prefix elements: literals, identifiers, grouped expressions
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        // Decide what kind of prefix element it is based on current_token
        match self.current_token.clone() {
            // Clone for matching
            Token::Number(value) => {
                self.next_token(); // Consume the number token
                Ok(Expression::NumberLiteral(value))
            }
            Token::Identifier(name) => {
                self.next_token(); // Consume the identifier token
                Ok(Expression::Variable(name))
            }
            Token::LParen => {
                // Delegate, DO NOT consume LParen here.
                // parse_grouped_expression handles consuming the matching RParen.
                self.next_token(); // Consume the '(' BEFORE calling the helper
                self.parse_grouped_expression() // Returns result, current_token is after ')'
            }
            // Add unary operators later (e.g., Token::Minus => ...)
            _ => Err(ParseError::UnexpectedToken {
                expected: "number, identifier, or '('".to_string(),
                found: self.current_token.clone(),
            }),
        }
        // REMOVE the blanket self.next_token() call from here
    }

    fn parse_grouped_expression(&mut self) -> ParseResult<Expression> {
        // LParen was consumed by the caller (parse_prefix)
        let expr = self.parse_expression(0)?;
        // Current token should now be RParen

        // Expect and consume ')'
        self.expect_and_consume(Token::RParen)?; // Handles check and consumption

        Ok(expr)
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
}

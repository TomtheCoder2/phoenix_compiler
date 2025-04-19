// src/parser.rs

use crate::ast::{BinaryOperator, Expression, NumberType};
use crate::lexer::Lexer;
use crate::token::Token;
use std::fmt;

// --- ParseError ---
// Add specific errors if needed, e.g., for expected '=' or 'in'
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken { expected: String, found: Token },
    // InvalidNumber(String), // Lexer seems to handle this now
    EndOfInput,
    // UnknownOperator(Token), // Not strictly needed if precedence handles it
    ExpectedExpression,
    ExpectedIdentifier,   // Added
    ExpectedToken(Token), // Added for specific tokens like = or in
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken { expected, found } => {
                write!(f, "Expected {}, found {:?}", expected, found)
            }
            ParseError::EndOfInput => write!(f, "Unexpected end of input"),
            ParseError::ExpectedExpression => write!(f, "Expected an expression"),
            ParseError::ExpectedIdentifier => write!(f, "Expected an identifier"),
            ParseError::ExpectedToken(token) => write!(f, "Expected token {:?}", token),
        }
    }
}

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
    // Using peek_token helps simplify some parsing logic, especially for operators vs function calls etc.
    // Let's add it back now.
    peek_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut parser = Parser {
            lexer,
            current_token: Token::Eof, // Placeholder
            peek_token: Token::Eof,    // Placeholder
        };
        // Load first two tokens
        parser.next_token();
        parser.next_token();
        parser
    }

    // Advance tokens using peek
    fn next_token(&mut self) {
        self.current_token = self.peek_token.clone(); // Shift peek to current
        self.peek_token = self.lexer.next_token(); // Get next from lexer
    }

    // Helper to check current token and advance if it matches
    fn expect_and_consume(&mut self, expected: Token) -> ParseResult<()> {
        if self.current_token == expected {
            self.next_token();
            Ok(())
        } else {
            Err(ParseError::ExpectedToken(expected))
        }
    }

    // Main entry point
    pub fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_expression_recursive(0)
    }

    // Pratt parsing recursive function (minor changes might be needed if precedence interacts with new forms)
    fn parse_expression_recursive(&mut self, min_precedence: u8) -> ParseResult<Expression> {
        // Parse prefix expression (number, identifier, '(', 'let')
        let mut left = self.parse_prefix()?;

        // Loop for infix operators
        while min_precedence < precedence(&self.current_token) // Check current_token now
            && token_to_binary_op(&self.current_token).is_some()
        {
            let op = token_to_binary_op(&self.current_token).unwrap(); // We know it's Some
            let current_precedence = precedence(&self.current_token);
            self.next_token(); // Consume the operator

            let right = self.parse_expression_recursive(current_precedence)?;

            left = Expression::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    // Updated to handle identifiers, 'let', and '('
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        let result = match self.current_token.clone() {
            // Clone for matching, advance later
            Token::Number(value) => Ok(Expression::NumberLiteral(value)),
            Token::Identifier(name) => Ok(Expression::Variable(name)), // Identifier is a variable usage
            Token::LParen => self.parse_grouped_expression(),
            Token::Let => self.parse_let_expression(), // Delegate 'let' parsing
            _ => Err(ParseError::ExpectedExpression),
        };
        // Consume the token *after* matching or delegating, only if successful so far
        if result.is_ok() {
            self.next_token();
        }
        result
    }

    // Parses expressions within parentheses
    fn parse_grouped_expression(&mut self) -> ParseResult<Expression> {
        // Assumes current token is '('. We consume it in parse_prefix *after* this call succeeds.
        // self.next_token(); // Consume '(' - No, consumed by caller (parse_prefix) after success

        // Parse the inner expression
        let expr = self.parse_expression_recursive(0)?;

        // Expect ')'
        if self.current_token == Token::RParen {
            // self.next_token(); // Consume ')' - No, consumed by caller
            Ok(expr)
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "closing parenthesis ')'".to_string(),
                found: self.current_token.clone(),
            })
        }
    }

    // Parses a 'let name = value in body' expression
    fn parse_let_expression(&mut self) -> ParseResult<Expression> {
        // Assumes current token is 'Let'. Consumed by caller (parse_prefix) after success.
        // Expect Identifier
        self.next_token(); // Move past 'let' to the identifier
        let name = match &self.current_token {
            Token::Identifier(n) => n.clone(),
            _ => return Err(ParseError::ExpectedIdentifier),
        };
        self.next_token(); // Consume identifier

        // Expect '='
        self.expect_and_consume(Token::Assign)?; // Consumes '='

        // Parse the value expression
        let value = self.parse_expression_recursive(0)?;
        // After parsing value, current_token should be 'in'

        // Expect 'in'
        self.expect_and_consume(Token::In)?; // Consumes 'in'

        // Parse the body expression
        let body = self.parse_expression_recursive(0)?;
        // After parsing body, current_token is the token following the let expr.
        // The caller (parse_prefix) will consume this final token of the let expr.

        Ok(Expression::Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
        })
    }
}

// --- Update Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOperator::*, Expression::*};
    use crate::lexer::Lexer;
    // Use * for Expression variants too

    fn num(val: NumberType) -> Box<Expression> {
        Box::new(NumberLiteral(val))
    }
    fn var(name: &str) -> Box<Expression> {
        Box::new(Variable(name.to_string()))
    }
    fn bin_op(op: BinaryOperator, left: Box<Expression>, right: Box<Expression>) -> Expression {
        BinaryOp { op, left, right }
    }
    fn let_expr(name: &str, value: Box<Expression>, body: Box<Expression>) -> Expression {
        Let {
            name: name.to_string(),
            value,
            body,
        }
    }

    #[test]
    fn test_parse_variable() {
        let input = "my_var";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();
        assert_eq!(expr, Variable("my_var".to_string()));
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_simple_let_expression() {
        // let x = 5 in x
        let input = "let x = 5 in x";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();
        let expected = let_expr("x", num(5.0), var("x"));
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_let_with_binary_op_body() {
        // let y = 10 in y * 2
        let input = "let y = 10 in y * 2";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();
        let expected = let_expr(
            "y",
            num(10.0),
            Box::new(bin_op(Multiply, var("y"), num(2.0))),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_let_with_binary_op_value() {
        // let y = 10 * 2 in y + 1
        let input = "let y = 10 * 2 in y + 1";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();
        let expected = let_expr(
            "y",
            Box::new(bin_op(Multiply, num(10.0), num(2.0))),
            Box::new(bin_op(Add, var("y"), num(1.0))),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_nested_let_expression() {
        // let a = 1 in let b = a + 2 in b * a
        let input = "let a = 1 in let b = a + 2 in b * a";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();
        /* Expected AST:
           Let { name: "a", value: 1.0,
                 body: Let { name: "b", value: BinaryOp(Add, Variable("a"), 2.0),
                             body: BinaryOp(Multiply, Variable("b"), Variable("a")) } }
        */
        let expected = let_expr(
            "a",
            num(1.0),
            Box::new(let_expr(
                "b",
                Box::new(bin_op(Add, var("a"), num(2.0))),
                Box::new(bin_op(Multiply, var("b"), var("a"))),
            )),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_let_in_binary_op() {
        // (let x = 5 in x) + 1
        let input = "(let x = 5 in x) + 1";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();
        let expected = bin_op(Add, Box::new(let_expr("x", num(5.0), var("x"))), num(1.0));
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parse_error_let_missing_equals() {
        let input = "let x 5 in x";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_expression();
        assert!(result.is_err());
        // Error occurs in parse_let_expression when expecting '='
        assert_eq!(
            result.unwrap_err(),
            ParseError::ExpectedToken(Token::Assign)
        );
    }

    #[test]
    fn test_parse_error_let_missing_in() {
        let input = "let x = 5 x";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_expression();
        assert!(result.is_err());
        // Error occurs in parse_let_expression when expecting 'in'
        assert_eq!(result.unwrap_err(), ParseError::ExpectedToken(Token::In));
    }
}

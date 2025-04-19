// src/parser.rs

use crate::ast::{BinaryOperator, Expression, NumberType};
use crate::lexer::Lexer;
use crate::token::Token;
use std::fmt;

// --- ParseError and ParseResult remain the same ---
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken { expected: String, found: Token },
    InvalidNumber(String), // Maybe remove if lexer handles this? Keep for now.
    EndOfInput,
    UnknownOperator(Token), // Added for Pratt
    ExpectedExpression,     // Added for Pratt
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken { expected, found } => {
                write!(f, "Expected {}, found {:?}", expected, found)
            }
            ParseError::InvalidNumber(num_str) => write!(f, "Invalid number literal: {}", num_str),
            ParseError::EndOfInput => write!(f, "Unexpected end of input"),
            ParseError::UnknownOperator(token) => write!(f, "Unknown operator: {:?}", token),
            ParseError::ExpectedExpression => write!(f, "Expected an expression"),
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
    // peek_token: Token, // Still not strictly needed for this simple Pratt version
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut parser = Parser {
            lexer,
            current_token: Token::Eof, // Placeholder
        };
        parser.next_token(); // Load the first token
        parser
    }

    fn next_token(&mut self) {
        self.current_token = self.lexer.next_token();
    }

    // Main entry point - parses an expression with minimum precedence 0
    pub fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_expression_recursive(0) // Start with lowest precedence
    }

    // The core Pratt parsing function
    fn parse_expression_recursive(&mut self, min_precedence: u8) -> ParseResult<Expression> {
        // 1. Parse the left-hand side (prefix expression)
        let mut left = self.parse_prefix()?; // Handle numbers, parentheses, unary ops...

        // 2. Loop while the next token is an infix operator with sufficient precedence
        // Use a loop instead of just 'if' to handle left-associativity correctly (e.g., a - b + c)
        while min_precedence < precedence(&self.current_token) {
            // Check if the current token is a valid binary operator *before* consuming it
            let op = match token_to_binary_op(&self.current_token) {
                Some(op) => op,
                None => break, // If it's not an operator we handle infix, stop the loop
            };

            let current_precedence = precedence(&self.current_token);
            self.next_token(); // Consume the operator token

            // 3. Parse the right-hand side
            // Recursively call, passing the current operator's precedence.
            // For left-associativity (like +, -, *, /), the minimum precedence for the right side
            // is the *same* as the current operator. For right-associativity (like exponentiation '^'),
            // it would be current_precedence - 1. Our operators are left-associative.
            let right = self.parse_expression_recursive(current_precedence)?;

            // 4. Combine left and right into a new 'left' for the next iteration
            left = Expression::BinaryOp {
                op, // The operator we just parsed
                left: Box::new(left),
                right: Box::new(right),
            };
        } // Loop continues if the *next* token is another operator of sufficient precedence

        Ok(left) // Return the fully assembled expression
    }

    // Parses prefix elements: literals, parenthesized expressions, unary operators (later)
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        let prefix_token = self.current_token.clone(); // Clone for analysis
        self.next_token(); // Consume the token *after* cloning

        match prefix_token {
            Token::Number(value) => Ok(Expression::NumberLiteral(value)),
            Token::LParen => {
                // Parse the expression inside the parentheses
                let expr_inside = self.parse_expression_recursive(0)?; // Start with lowest precedence inside (

                // Expect a closing parenthesis
                if self.current_token == Token::RParen {
                    self.next_token(); // Consume the ')'
                    Ok(expr_inside) // Return the expression that was inside
                } else {
                    Err(ParseError::UnexpectedToken {
                        expected: "closing parenthesis ')'".to_string(),
                        found: self.current_token.clone(),
                    })
                }
            }
            // Handle unary minus later, e.g., Token::Minus => ...
            _ => Err(ParseError::ExpectedExpression), // More specific error if needed
        }
    }
}

// --- Update Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinaryOperator::*;
    use crate::lexer::Lexer;
    // Import operators for easier matching

    // Helper to create Boxed expressions concisely in tests
    fn num(val: NumberType) -> Box<Expression> {
        Box::new(Expression::NumberLiteral(val))
    }

    fn bin_op(op: BinaryOperator, left: Box<Expression>, right: Box<Expression>) -> Expression {
        Expression::BinaryOp { op, left, right }
    }

    #[test]
    fn test_parse_number_literal() {
        let input = "123.45";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        assert_eq!(expr, Expression::NumberLiteral(123.45));
        assert_eq!(parser.current_token, Token::Eof); // Ensure EOF
    }

    #[test]
    fn test_simple_addition() {
        let input = "1 + 2";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        assert_eq!(expr, bin_op(Add, num(1.0), num(2.0)));
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_precedence() {
        // Expect: 1 + (2 * 3)
        let input = "1 + 2 * 3";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        let expected = bin_op(
            Add,
            num(1.0),
            Box::new(bin_op(Multiply, num(2.0), num(3.0))),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_precedence_division_subtraction() {
        // Expect: (10 / 2) - 3
        let input = "10 / 2 - 3";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        let expected = bin_op(
            Subtract,
            Box::new(bin_op(Divide, num(10.0), num(2.0))),
            num(3.0),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_left_associativity() {
        // Expect: (1 - 2) + 3
        let input = "1 - 2 + 3";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        let expected = bin_op(
            Add,
            Box::new(bin_op(Subtract, num(1.0), num(2.0))),
            num(3.0),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parentheses() {
        // Expect: (1 + 2) * 3
        let input = "(1 + 2) * 3";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        let expected = bin_op(
            Multiply,
            Box::new(bin_op(Add, num(1.0), num(2.0))),
            num(3.0),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_nested_parentheses() {
        // Expect: ((1 + 2) * (3 - 4)) / 5
        let input = "((1 + 2) * (3 - 4)) / 5";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let expr = parser.parse_expression().unwrap();

        let expected = bin_op(
            Divide,
            Box::new(bin_op(
                Multiply,
                Box::new(bin_op(Add, num(1.0), num(2.0))),
                Box::new(bin_op(Subtract, num(3.0), num(4.0))),
            )),
            num(5.0),
        );
        assert_eq!(expr, expected);
        assert_eq!(parser.current_token, Token::Eof);
    }

    #[test]
    fn test_parse_error_unexpected_token() {
        let input = "1 * + 2"; // Invalid sequence
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_expression(); // 1 is parsed, * is consumed, then tries to parse prefix for '+'

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ParseError::ExpectedExpression); // Because '+' is not a valid start of an expression
    }

    #[test]
    fn test_parse_error_missing_paren() {
        let input = "(1 + 2";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_expression();

        assert!(result.is_err());
        // The error occurs when parse_prefix looks for ')' after parsing '1+2'
        assert_eq!(
            result.unwrap_err(),
            ParseError::UnexpectedToken {
                expected: "closing parenthesis ')'".to_string(),
                found: Token::Eof
            }
        );
    }

    #[test]
    fn test_parse_error_leading_operator() {
        let input = "* 1 + 2";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse_expression(); // Tries to parse prefix for '*'

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ParseError::ExpectedExpression);
    }
}

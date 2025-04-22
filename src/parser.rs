// src/parser.rs

// Use the new AST structures
use crate::ast::{
    BinaryOperator, ComparisonOperator, Expression, ExpressionKind, Program, Statement,
    StatementKind, TypeNode, TypeNodeKind, UnaryOperator,
};
use crate::lexer::Lexer;
use crate::location::{Location, Span};
use crate::parser::Precedence::Lowest;
use crate::token::{Token, TokenKind};
use std::fmt;

// --- ParseError --- (Adjust as needed)
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken {
        expected: String,
        found: TokenKind,
        loc: Location,
    },
    EndOfInput {
        loc: Location,
    },
    ExpectedExpression {
        loc: Location,
    },
    ExpectedIdentifier {
        loc: Location,
    },
    ExpectedToken {
        expected: TokenKind,
        loc: Location,
    },
    InvalidTypeAnnotation {
        type_name: String,
        loc: Location,
    },
    InvalidAssignmentTarget {
        target: String,
        loc: Location,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken {
                expected,
                found,
                loc,
            } => {
                write!(
                    f,
                    "Parse Error at {}: Expected {}, found {:?}",
                    loc, expected, found
                )
            }
            ParseError::EndOfInput { loc } => {
                write!(f, "Parse Error at {}: Unexpected end of input", loc)
            }
            ParseError::ExpectedExpression { loc } => {
                write!(f, "Parse Error at {}: Expected an expression", loc)
            }
            ParseError::ExpectedIdentifier { loc } => {
                write!(f, "Parse Error at {}: Expected an identifier", loc)
            }
            ParseError::ExpectedToken { expected, loc } => {
                write!(f, "Parse Error at {}: Expected token {:?}", loc, expected)
            }
            ParseError::InvalidTypeAnnotation { type_name, loc } => {
                write!(
                    f,
                    "Parse Error at {}: Invalid type annotation '{}'",
                    loc, type_name
                )
            }
            ParseError::InvalidAssignmentTarget { target, loc } => {
                write!(
                    f,
                    "Parse Error at {}: Invalid assignment target (must be variable or index access)'{}'",
                    loc, target
                )
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
    Index,       // array[index] // Added (highest precedence)
}

// Map tokens to precedence levels
fn token_precedence(token: &TokenKind) -> Precedence {
    match token {
        TokenKind::Assign => Precedence::Assign,
        TokenKind::Equal | TokenKind::NotEqual => Precedence::Equals,
        TokenKind::LessThan
        | TokenKind::GreaterThan
        | TokenKind::LessEqual
        | TokenKind::GreaterEqual => Precedence::LessGreater,
        TokenKind::Plus | TokenKind::Minus => Precedence::Sum, // Infix minus
        TokenKind::Star | TokenKind::Slash => Precedence::Product,
        TokenKind::LParen => Precedence::Call,
        TokenKind::LBracket => Precedence::Index, // Added
        _ => Lowest,
    }
}

// Map tokens to AST unary operators (used in parse_prefix)
fn token_to_unary_op(token: &TokenKind) -> Option<UnaryOperator> {
    match token {
        TokenKind::Minus => Some(UnaryOperator::Negate),
        TokenKind::Bang => Some(UnaryOperator::Not),
        _ => None,
    }
}

// Helper to convert TokenKind to BinaryOperator
fn token_to_binary_op(token: &TokenKind) -> Option<BinaryOperator> {
    match token {
        TokenKind::Plus => Some(BinaryOperator::Add),
        TokenKind::Minus => Some(BinaryOperator::Subtract),
        TokenKind::Star => Some(BinaryOperator::Multiply),
        TokenKind::Slash => Some(BinaryOperator::Divide),
        _ => None,
    }
}

// Map tokens to AST operators (split binary and comparison)
fn token_to_comparison_op(token: &TokenKind) -> Option<ComparisonOperator> {
    match token {
        TokenKind::LessThan => Some(ComparisonOperator::LessThan),
        TokenKind::GreaterThan => Some(ComparisonOperator::GreaterThan),
        TokenKind::Equal => Some(ComparisonOperator::Equal),
        TokenKind::NotEqual => Some(ComparisonOperator::NotEqual),
        TokenKind::LessEqual => Some(ComparisonOperator::LessEqual),
        TokenKind::GreaterEqual => Some(ComparisonOperator::GreaterEqual),
        _ => None,
    }
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    pub(crate) current_token: Token,
    peek_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        // Need dummy locations for initial Eof tokens
        let dummy_loc = Location::default();
        let eof_token = Token {
            kind: TokenKind::Eof,
            loc: dummy_loc.clone(),
        };
        let mut parser = Parser {
            lexer,
            current_token: eof_token.clone(),
            peek_token: eof_token,
        };
        parser.next_token();
        parser.next_token();
        parser
    }

    fn next_token(&mut self) {
        self.current_token = self.peek_token.clone();
        self.peek_token = self.lexer.next_token();
    }

    // Helper to check KIND and consume
    fn expect_and_consume_kind(&mut self, expected_kind: TokenKind) -> ParseResult<Location> {
        if self.current_token.kind == expected_kind {
            let loc = self.current_token.loc.clone(); // Get loc before consuming
            self.next_token();
            Ok(loc)
        } else {
            Err(ParseError::ExpectedToken {
                expected: expected_kind,
                loc: self.current_token.loc.clone(),
            })
        }
    }

    // --- Main Parsing Logic: Parse the whole program ---

    // --- Main Parsing Logic ---
    pub fn parse_program(&mut self) -> Result<Program, Vec<ParseError>> {
        let start_loc = self.current_token.loc.clone(); // Location of first token
        let mut statements = Vec::new();
        let mut errors = Vec::new();

        while self.current_token.kind != TokenKind::Eof {
            match self.parse_statement() {
                Ok(stmt) => statements.push(stmt),
                Err(err) => {
                    errors.push(err);
                    // Basic error recovery
                    while self.current_token.kind != TokenKind::Eof
                        && self.current_token.kind != TokenKind::RBrace
                        && self.current_token.kind != TokenKind::Eof
                    {
                        // Also recover on RBrace
                        self.next_token();
                    }
                    if self.current_token.kind == TokenKind::Semicolon
                        || self.current_token.kind == TokenKind::RBrace
                    {
                        self.next_token();
                    }
                }
            }
        }
        let end_loc = self.current_token.loc.clone(); // Location of EOF token
        let program_span = Span::from_locations(start_loc, end_loc);
        // ... return Ok(Program) or Err(errors) ...
        if errors.is_empty() {
            Ok(Program {
                statements,
                span: program_span,
            })
        } else {
            Err(errors)
        }
    }

    // --- Statement Parsing (Return Statement struct) ---
    fn parse_statement(&mut self) -> ParseResult<Statement> {
        let start_loc = self.current_token.loc.clone(); // Remember start
                                                        // Dispatch based on current_token.kind
        let kind_result = match self.current_token.kind {
            TokenKind::Let => self.parse_let_or_var_statement_kind(false),
            TokenKind::Var => self.parse_let_or_var_statement_kind(true),
            TokenKind::Fun => self.parse_function_definition_kind(),
            TokenKind::If => self.parse_if_statement_kind(),
            TokenKind::While => self.parse_while_statement_kind(),
            TokenKind::For => self.parse_for_statement_kind(),
            TokenKind::Return => self.parse_return_statement_kind(),
            // If none of the above, assume it's an expression statement
            _ => self.parse_expression_statement_kind(),
        };

        // Combine start location with end location (after parsing statement)
        // Need end location - maybe parse_ helpers should return it?
        // Simpler: Use peek_token's location as approximate end? Risky.
        // Let's pass start_loc down and have helpers return Span or end Loc.

        // Rethink: Each parse_* function returns Result<StatementKind, ParseError>
        // And also consumes tokens. We get end loc *after* the call.
        let kind = kind_result?;
        let end_loc = self.current_token.loc.clone(); // Location *after* statement parsed
        let span = Span::from_locations(start_loc, end_loc); // Approximate span
        Ok(Statement { kind, span })

        // Need to refactor parse_* helpers to return Kind, not Statement struct
    }

    // Parses: for ( [INIT_EXPR]? ; [COND_EXPR]? ; [INCR_EXPR]? ) { BODY }
    // No trailing semicolon expected/consumed.
    fn parse_for_statement_kind(&mut self) -> ParseResult<StatementKind> {
        self.next_token(); // Consume 'for'
        self.expect_and_consume_kind(TokenKind::LParen)?;

        // 1. Parse Initializer (Optional Expression before first ';')
        let initializer = if self.current_token.kind == TokenKind::Semicolon {
            None // No initializer expression
        } else {
            // Parse expression, box it
            match self.parse_statement() {
                Ok(stmt) => Some(Box::new(stmt)), // Wrap in Box
                Err(err) => {
                    return Err(err); // Propagate error
                }
            }
        };
        // self.expect_and_consume_kind(TokenKind::Semicolon)?; // Consume first ';'

        // 2. Parse Condition (Optional Expression before second ';')
        let condition = if self.current_token.kind == TokenKind::Semicolon {
            None // No condition expression (defaults to true)
        } else {
            Some(self.parse_expression(Precedence::Lowest)?)
        };
        self.expect_and_consume_kind(TokenKind::Semicolon)?; // Consume second ';'

        // 3. Parse Increment (Optional Expression before ')')
        let increment = if self.current_token.kind == TokenKind::RParen {
            None // No increment expression
        } else {
            // Parse expression, box it
            Some(self.parse_expression(Precedence::Lowest)?)
        };
        self.expect_and_consume_kind(TokenKind::RParen)?; // Consume ')'

        // 4. Parse Body
        self.expect_and_consume_kind(TokenKind::LBrace)?;
        let body = self.parse_block_statements()?; // Consumes '}'

        Ok(StatementKind::ForStmt {
            initializer,
            condition,
            increment,
            body,
        })
    }

    // Parses: return [EXPRESSION]? ;
    fn parse_return_statement_kind(&mut self) -> ParseResult<StatementKind> {
        let _start_token_loc = self.current_token.loc.clone();
        self.next_token(); // Consume 'return'
        let value: Option<Expression> = if self.current_token.kind != TokenKind::Semicolon {
            Some(self.parse_expression(Precedence::Lowest)?) // Parse expr (returns Expression struct)
        } else {
            None
        };
        self.expect_and_consume_kind(TokenKind::Semicolon)?; // Expect/consume ';'
        Ok(StatementKind::ReturnStmt { value })
    }

    // Parses: while ( CONDITION ) { BODY }
    // No trailing semicolon expected/consumed.
    fn parse_while_statement_kind(&mut self) -> ParseResult<StatementKind> {
        self.next_token(); // Consume 'while'

        self.expect_and_consume_kind(TokenKind::LParen)?;
        let condition = self.parse_expression(Lowest)?;
        self.expect_and_consume_kind(TokenKind::RParen)?;

        self.expect_and_consume_kind(TokenKind::LBrace)?;
        let body = self.parse_block_statements()?; // Consumes '}'

        Ok(StatementKind::WhileStmt { condition, body })
    }

    // Parses: if ( CONDITION ) { THEN_BRANCH } [ else { ELSE_BRANCH } ]
    // Note: No trailing semicolon expected/consumed here.
    fn parse_if_statement_kind(&mut self) -> ParseResult<StatementKind> {
        self.next_token(); // Consume 'if'
        self.expect_and_consume_kind(TokenKind::LParen)?;
        let condition = self.parse_expression(Lowest)?;
        self.expect_and_consume_kind(TokenKind::RParen)?;
        self.expect_and_consume_kind(TokenKind::LBrace)?; // Expect '{'
        let then_branch = self.parse_block_statements()?;

        // --- Optional Else ---
        let else_branch: Option<Program>;
        if self.current_token.kind == TokenKind::Else {
            self.next_token(); // Consume 'else'
            self.expect_and_consume_kind(TokenKind::LBrace)?;
            else_branch = Some(self.parse_block_statements()?); // Consumes '}'
        } else {
            else_branch = None; // No else branch found
        }

        Ok(StatementKind::IfStmt {
            condition, // condition is an Expression now, not Boxed
            then_branch,
            else_branch,
        })
    }

    // Parses: fun IDENT ( [PARAM_LIST]? ) [: TYPE]? { PROGRAM } ;?
    fn parse_function_definition_kind(&mut self) -> ParseResult<StatementKind> {
        self.next_token(); // Consume 'fun'
        let name = match &self.current_token.kind {
            TokenKind::Identifier(n) => n.clone(),
            _ => {
                return Err(ParseError::ExpectedIdentifier {
                    loc: self.current_token.loc.clone(),
                })
            }
        };
        self.next_token(); // Consume name
        self.expect_and_consume_kind(TokenKind::LParen)?;
        let params = self.parse_parameter_list()?; // Now handles types

        // Optional return type annotation
        let return_type_ann: Option<TypeNode> = if self.current_token.kind == TokenKind::Colon {
            self.next_token(); // Consume ':'
            Some(self.parse_type_annotation()?) // Parse type annotation
        } else {
            None
        };

        self.expect_and_consume_kind(TokenKind::LBrace)?;
        let body = self.parse_block_statements()?;
        if self.current_token.kind == TokenKind::Semicolon {
            self.next_token();
        } // Optional semi

        Ok(StatementKind::FunctionDef {
            name,
            params,
            return_type_ann,
            body,
        })
    }

    // Parses: IDENT [: TYPE]? [, IDENT [: TYPE]?]*
    fn parse_parameter_list(&mut self) -> ParseResult<Vec<(String, Option<TypeNode>)>> {
        let mut params = Vec::new();
        if self.current_token.kind == TokenKind::RParen {
            self.next_token();
            return Ok(params);
        }

        // Parse first param: IDENT [: TYPE]?
        let (first_name, first_type) = self.parse_single_param()?;
        params.push((first_name, first_type));

        while self.current_token.kind == TokenKind::Comma {
            self.next_token(); // Consume ','
            let (name, type_ann) = self.parse_single_param()?;
            params.push((name, type_ann));
        }

        self.expect_and_consume_kind(TokenKind::RParen)?;
        Ok(params)
    }

    // Helper to parse a single "IDENT [: TYPE]?"
    fn parse_single_param(&mut self) -> ParseResult<(String, Option<TypeNode>)> {
        let name = match &self.current_token.kind {
            TokenKind::Identifier(n) => n.clone(),
            _ => {
                return Err(ParseError::ExpectedIdentifier {
                    loc: self.current_token.loc.clone(),
                })
            }
        };
        self.next_token(); // Consume identifier

        let type_ann: Option<TypeNode> = if self.current_token.kind == TokenKind::Colon {
            self.next_token(); // Consume ':'
            Some(self.parse_type_annotation()?) // Parse type annotation
        } else {
            None
        };

        Ok((name, type_ann))
    }

    // Helper to parse statements within { ... }
    fn parse_block_statements(&mut self) -> ParseResult<Program> {
        let mut statements = Vec::new();
        let start_loc = self.current_token.loc.clone();
        // Note: Does not collect errors like top-level parse_program, stops on first error.
        // Could be enhanced later.

        // Loop until RBrace or EOF
        while self.current_token.kind != TokenKind::RBrace
            && self.current_token.kind != TokenKind::Eof
        {
            let stmt = self.parse_statement()?; // Parse one statement
            statements.push(stmt);
        }

        // Expect closing '}'
        self.expect_and_consume_kind(TokenKind::RBrace)?;
        let end_loc = self.current_token.loc.clone(); // Location of '}'
        let span = Span::from_locations(start_loc, end_loc); // Create span
        Ok(Program { statements, span })
    }

    // Combined parser for let/var: let/var IDENT [: TYPE]? = EXPRESSION ;
    fn parse_let_or_var_statement_kind(&mut self, is_mutable: bool) -> ParseResult<StatementKind> {
        self.next_token(); // Consume 'let' or 'var'

        let name = match &self.current_token.kind {
            TokenKind::Identifier(n) => n.clone(),
            _ => {
                return Err(ParseError::ExpectedIdentifier {
                    loc: self.current_token.loc.clone(),
                })
            }
        };
        self.next_token(); // Consume identifier

        // --- Optional Type Annotation ---
        let type_ann: Option<TypeNode> = if self.current_token.kind == TokenKind::Colon {
            self.next_token(); // Consume ':'
            Some(self.parse_type_annotation()?) // Parse type annotation
        } else {
            None
        };

        self.expect_and_consume_kind(TokenKind::Assign)?; // Expect and consume initial '='
        let value_expr = self.parse_expression(Lowest)?;
        self.expect_and_consume_kind(TokenKind::Semicolon)?;

        if is_mutable {
            Ok(StatementKind::VarBinding {
                name,
                type_ann,
                value: value_expr,
            })
        } else {
            Ok(StatementKind::LetBinding {
                name,
                type_ann,
                value: value_expr,
            })
        }
    }

    // Parses: EXPRESSION ;
    fn parse_expression_statement_kind(&mut self) -> ParseResult<StatementKind> {
        let expr = self.parse_expression(Lowest)?; // Use enum precedence
        self.expect_and_consume_kind(TokenKind::Semicolon)?;
        Ok(StatementKind::ExpressionStmt(expr))
    }

    // --- Expression Parsing (Pratt Parser - mostly unchanged internally) ---

    // --- Expression Parsing (Updated Pratt Parser) ---
    fn parse_expression(&mut self, precedence: Precedence) -> ParseResult<Expression> {
        let start_loc = self.current_token.loc.clone();
        // Parse prefix expression
        let mut left = self.parse_prefix()?;

        // Loop while next token is infix and has higher precedence
        while precedence < token_precedence(&self.current_token.kind) {
            let infix_token = self.current_token.clone(); // Keep whole token for loc/kind
                                                          // Check CURRENT token now
                                                          // Check if current token is a binary/comparison operator OR function call start
            match infix_token.kind {
                // Arithmetic Binary Ops
                TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash => {
                    let op = token_to_binary_op(&infix_token.kind).unwrap();
                    let current_precedence = token_precedence(&infix_token.kind);
                    self.next_token(); // Consume operator
                    let right = self.parse_expression(current_precedence)?; // Returns Expression struct
                    let combined_span = left.span.combine(&right.span); // Combine spans
                    left = Expression::new(
                        // Create new wrapper struct
                        ExpressionKind::BinaryOp {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        combined_span,
                    );
                }
                // Comparison Ops
                TokenKind::LessThan
                | TokenKind::GreaterThan
                | TokenKind::Equal
                | TokenKind::NotEqual
                | TokenKind::LessEqual
                | TokenKind::GreaterEqual => {
                    let op = token_to_comparison_op(&self.current_token.kind).unwrap(); // Known to be Some
                    let current_precedence = token_precedence(&self.current_token.kind);
                    self.next_token(); // Consume operator
                    let right = self.parse_expression(current_precedence)?;
                    let span = left.span.combine(&right.span); // Combine spans
                    left = Expression::new(
                        ExpressionKind::ComparisonOp {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                }
                // Function Call
                TokenKind::LParen => {
                    // Need to check if 'left' was actually a Variable
                    let func_name = match left.kind {
                        ExpressionKind::Variable(ref name) => name,
                        _ => {
                            return Err(ParseError::UnexpectedToken {
                                expected: "function name".to_string(),
                                found: self.current_token.kind.clone(),
                                loc: self.current_token.loc.clone(),
                            })
                        }
                    };
                    self.next_token(); // Consume '('
                    let args = self.parse_argument_list()?; // Returns vec of expressions
                    let span = left.span.combine(&args.last().unwrap_or(&left).span); // Combine spans
                    left = Expression::new(
                        ExpressionKind::FunctionCall {
                            name: func_name.clone(),
                            args,
                        },
                        span,
                    );
                }
                TokenKind::Assign => {
                    // Check if 'left' is a valid L-value (Variable or IndexAccess)
                    match left.kind {
                        ExpressionKind::Variable(_) | ExpressionKind::IndexAccess {..} => {
                            // OK, proceed with assignment
                        }
                        _ => {
                            // Invalid target
                            return Err(ParseError::InvalidAssignmentTarget { target: format!("{:?}", left), loc: left.span.start.clone() });
                        }
                    };

                    let current_precedence = token_precedence(&infix_token.kind);
                    self.next_token(); // Consume '='

                    // Parse RHS (handle right-associativity carefully if needed)
                    let value = self.parse_expression(current_precedence)?; // Use same precedence for now

                    let combined_span = left.span.combine(&value.span);
                    // Create Assignment node with the *entire* left expression as target
                    left = Expression::new(
                        ExpressionKind::Assignment { target: Box::new(left), value: Box::new(value) },
                        combined_span
                    );
                } // End Assign case
                TokenKind::LBracket => {
                    // --- Index Operator ---
                    self.next_token(); // Consume '['
                    let index_expr = self.parse_expression(Precedence::Lowest)?; // Parse index
                    let end_loc = self.expect_and_consume_kind(TokenKind::RBracket)?; // Consume ']'
                    let span = Span::from_locations(left.span.start.clone(), end_loc); // Span from target start to ']'
                    left = Expression::new(
                        ExpressionKind::IndexAccess {
                            target: Box::new(left),
                            index: Box::new(index_expr),
                        },
                        span,
                    );
                }
                // Not an operator we handle infix at this precedence
                _ => break,
            }
        } // End loop
        Ok(left)
    }

    // Parses prefix elements: literals, identifiers, grouped expr, maybe unary later
    // Parses prefix elements. Now handles 'if' as a prefix for an expression.
    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        let token = self.current_token.clone();
        let start_loc = token.loc.clone();
        let span = Span::single(start_loc.clone()); // Initial span assumption

        match token.kind {
            // Literals
            TokenKind::FloatNum(v) => {
                self.next_token();
                Ok(Expression::new(ExpressionKind::FloatLiteral(v), span))
            }
            TokenKind::IntNum(v) => {
                self.next_token();
                Ok(Expression::new(ExpressionKind::IntLiteral(v), span))
            }
            TokenKind::BoolLiteral(v) => {
                self.next_token();
                Ok(Expression::new(ExpressionKind::BoolLiteral(v), span))
            }
            TokenKind::StringLiteral(s) => {
                self.next_token();
                Ok(Expression::new(ExpressionKind::StringLiteral(s), span))
            }
            // Variable
            TokenKind::Identifier(n) => {
                self.next_token();
                Ok(Expression::new(ExpressionKind::Variable(n), span))
            }
            // Grouping
            TokenKind::LParen => {self.next_token(); self.parse_grouped_expression(start_loc) }, // Returns full Expression struct
            // [
            TokenKind::LBracket => self.parse_vector_literal(start_loc),
            // If Expression
            TokenKind::If => self.parse_if_expression(), // Returns full Expression struct
            // Block Expression
            TokenKind::LBrace => self.parse_block_expression(), // Returns full Expression struct
            // Unary Operators
            TokenKind::Minus | TokenKind::Bang => {
                self.next_token();
                let op = token_to_unary_op(&token.kind).unwrap();
                let operand = self.parse_expression(Precedence::Prefix)?;
                let op_span = Span::from_locations(start_loc, operand.span.end.clone()); // Combine spans
                Ok(Expression::new(
                    ExpressionKind::UnaryOp {
                        op,
                        operand: Box::new(operand),
                    },
                    op_span,
                ))
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: token.kind.clone(),
                loc: token.loc.clone(),
            }),
        }
    }

    // Parses VectorLiteral: [ [EXPR [, EXPR]*]? ]
    fn parse_vector_literal(&mut self, start_loc: Location) -> ParseResult<Expression> {
        self.next_token(); // Consume '['

        let mut elements = Vec::new();
        // Handle empty vector []
        if self.current_token.kind != TokenKind::RBracket {
            // Parse first element
            elements.push(self.parse_expression(Precedence::Lowest)?);
            // Parse remaining elements
            while self.current_token.kind == TokenKind::Comma {
                self.next_token(); // Consume ','
                                   // Allow trailing comma? If so, check for ']' before parsing expr.
                if self.current_token.kind == TokenKind::RBracket {
                    break;
                }
                elements.push(self.parse_expression(Precedence::Lowest)?);
            }
        }

        let end_loc = self.expect_and_consume_kind(TokenKind::RBracket)?; // Consume ']'
        let span = Span::from_locations(start_loc, end_loc);
        Ok(Expression::new(
            ExpressionKind::VectorLiteral { elements },
            span,
        ))
    }

    // Parses: if ( CONDITION ) { THEN_BRANCH } else { ELSE_BRANCH }
    // Consumes all tokens from 'if' to the final '}' of the else block.
    // Parses IfExpr: if ( CONDITION ) THEN_EXPR else ELSE_EXPR
    // Branches are now parsed as expressions (often block expressions)
    fn parse_if_expression(&mut self) -> ParseResult<Expression> {
        self.next_token(); // Consume 'if'
        self.expect_and_consume_kind(TokenKind::LParen)?;
        let condition = self.parse_expression(Lowest)?;
        self.expect_and_consume_kind(TokenKind::RParen)?;

        // Expect THEN branch expression (could be a block expression)
        let then_branch = self.parse_expression(Lowest)?;

        // Expect 'else'
        self.expect_and_consume_kind(TokenKind::Else)?;

        // Expect ELSE branch expression (could be a block expression)
        let else_branch = self.parse_expression(Lowest)?;

        let start_loc = condition.span.start.clone();
        let end_loc = else_branch.span.end.clone(); // End of the last expression
        Ok(Expression::new(
            ExpressionKind::IfExpr {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            Span::from_locations(start_loc, end_loc),
        ))
    }
    // Helper (optional, simple version)
    fn is_potential_statement_start(&self) -> bool {
        // Very basic check. Needs refinement.
        matches!(
            self.current_token.kind,
            TokenKind::Let | TokenKind::Var | TokenKind::Fun | TokenKind::If
        )
    }

    // Parses BlockExpr: { [ STMT; ]* [ EXPR ]? }
    // Consumes initial '{' and final '}'.
    fn parse_block_expression(&mut self) -> ParseResult<Expression> {
        let start_loc = self.current_token.loc.clone(); // Location of '{'
        self.next_token(); // Consume '{'

        let mut statements: Vec<Statement> = Vec::new();
        let mut final_expression: Option<Box<Expression>> = None;

        while self.current_token.kind != TokenKind::RBrace
            && self.current_token.kind != TokenKind::Eof
        {
            // Try parsing a statement first
            if self.peek_token.kind == TokenKind::Semicolon
                || (
                    // Heuristic: Look for tokens likely starting a statement vs expression
                    // e.g., Let, Var, If often start statements. Others *might* be exprs.
                    // This is tricky without full lookahead or backtracking.
                    // Simpler approach: If parsing statement fails, *try* parsing as final expr.
                    // Let's try: parse everything as statement candidate first.
                    matches!(
                        self.current_token.kind,
                        TokenKind::Let | TokenKind::Var | TokenKind::Fun | TokenKind::If
                    )
                )
                || self.is_potential_statement_start()
            // Add helper if needed
            {
                // Assume it's a statement
                let stmt = self.parse_statement()?; // parse_statement expects semi or handles blocks
                statements.push(stmt);
            } else {
                // Not obviously a statement, assume it's the final expression
                // If there's another token after this that's not '}', error?
                let expr = self.parse_expression(Lowest)?;
                // Check if next token is '}' - if so, this is the final expression
                if self.current_token.kind == TokenKind::RBrace {
                    final_expression = Some(Box::new(expr));
                    // Don't break yet, let the loop condition handle RBrace
                } else {
                    // If it's not RBrace, it MUST be a semicolon for ExpressionStmt
                    self.expect_and_consume_kind(TokenKind::Semicolon)?;
                    let span = Span::from_locations(
                        statements
                            .last()
                            .map_or(start_loc.clone(), |s| s.span.start.clone()),
                        expr.span.end.clone(),
                    );
                    statements.push(Statement {
                        kind: StatementKind::ExpressionStmt(expr),
                        span,
                    });
                }
            }

            // // Alternative logic: Check for semicolon explicitly
            // let expr_or_stmt_result = self.parse_expression_or_statement_in_block();
            // match expr_or_stmt_result {
            //    Ok(Either::Left(stmt)) => statements.push(stmt),
            //    Ok(Either::Right(expr)) => { final_expression = Some(Box::new(expr)); break; } // Found final expr
            //    Err(e) => return Err(e),
            // }
        }

        self.expect_and_consume_kind(TokenKind::RBrace)?; // Consume '}'

        let span = Span::from_locations(
            statements
                .first()
                .map_or(start_loc.clone(), |s| s.span.start.clone()),
            self.current_token.loc.clone(),
        );
        Ok(Expression::new(
            ExpressionKind::Block {
                statements,
                final_expression,
            },
            span,
        ))
    }

    // Parses `( EXPR )` - consumes the initial `(` and final `)`
    fn parse_grouped_expression(&mut self, lparen_loc: Location) -> ParseResult<Expression> {
        // LParen already consumed by caller
        let expr = self.parse_expression(Precedence::Lowest)?; // Get inner Expression struct
        let rparen_loc = self.expect_and_consume_kind(TokenKind::RParen)?; // Get ')' location
        let span = Span::from_locations(lparen_loc, rparen_loc); // Span from ( to )
                                                                 // Return the inner expression, but adjust its span to cover the parentheses
        Ok(Expression::new(expr.kind, span)) // Overwrite inner span
    }

    // Helper to parse EXPRESSION [, EXPRESSION]* within parentheses for function calls
    fn parse_argument_list(&mut self) -> ParseResult<Vec<Expression>> {
        let mut args = Vec::new();

        // Handle empty arg list: foo()
        if self.current_token.kind == TokenKind::RParen {
            self.next_token(); // Consume ')'
            return Ok(args);
        }

        // Parse first argument expression
        args.push(self.parse_expression(Lowest)?);

        // Parse subsequent arguments (comma followed by expression)
        while self.current_token.kind == TokenKind::Comma {
            self.next_token(); // Consume ','
            args.push(self.parse_expression(Lowest)?);
        }

        // Expect closing ')'
        self.expect_and_consume_kind(TokenKind::RParen)?;

        Ok(args)
    }

    // --- Type Annotation Parsing ---
    // Parses a type annotation like "int", "float", "vec<int>"
    fn parse_type_annotation(&mut self) -> ParseResult<TypeNode> {
        let start_loc = self.current_token.loc.clone();
        match self.current_token.kind.clone() {
            // Check for "vec" identifier first
            TokenKind::Identifier(ref name) if name == "vec" => {
                self.next_token(); // Consume "vec"
                self.expect_and_consume_kind(TokenKind::LessThan)?; // Expect '<'
                                                                    // Recursively parse the inner type node
                let element_type = self.parse_type_annotation()?;
                let end_loc = self.expect_and_consume_kind(TokenKind::GreaterThan)?; // Expect '>'
                let span = Span::from_locations(start_loc, end_loc);
                Ok(TypeNode {
                    kind: TypeNodeKind::Vector(Box::new(element_type)),
                    span,
                })
            }
            // Simple type name
            TokenKind::Identifier(name) => {
                // TODO: Validate if name is actually a known type? Typechecker does this.
                self.next_token(); // Consume identifier
                let span = Span::single(start_loc);
                Ok(TypeNode {
                    kind: TypeNodeKind::Simple(name),
                    span,
                })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "type name or vec<...>".to_string(),
                found: self.current_token.kind.clone(),
                loc: start_loc,
            }),
        }
    }

    // `parse_let_expression` is removed, replaced by `parse_let_statement`
}

// --- Update Tests ---

//! Wisp syntax: lexer, parser and AST.

pub mod ast;
pub mod diag;
pub mod lexer;
pub mod parser;

pub use diag::{Diag, Span};

/// Lex and parse a whole source file.
pub fn parse(src: &str) -> Result<ast::Program, Diag> {
    let toks = lexer::lex(src)?;
    parser::Parser::new(toks).program()
}

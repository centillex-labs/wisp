//! Abstract syntax tree.

use crate::diag::Span;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    MatMul,
    Div,
    LeftDiv,
    Pow,
    ElMul,
    ElDiv,
    ElPow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Plus,
    Not,
}

#[derive(Clone, Debug)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub enum Post {
    Call(Vec<Arg>),
    Index(Vec<Arg>),
    CellIndex(Vec<Arg>),
    Field(String),
    DynField(Box<Expr>),
    Transpose,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Num(f64),
    Str(String),
    Ident(String),
    Bool(bool),
    /// `last` inside an index.
    Last,
    /// `:` inside an index.
    Colon,
    /// `~` placeholder (only valid as a multi-assignment target).
    Tilde,
    Matrix(Vec<Vec<Expr>>),
    Cell(Vec<Vec<Expr>>),
    Lambda(Rc<Vec<String>>, Rc<Expr>),
    FuncRef(String),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Range(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Postfix(Box<Expr>, Post),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
}

/// An assignment target: a variable followed by fields, indices and cell indices.
#[derive(Clone, Debug)]
pub struct LValue {
    pub base: String,
    pub path: Vec<Post>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Expr(Expr),
    Assign(LValue, AssignOp, Expr),
    Multi(Vec<Option<LValue>>, Expr),
    If(Vec<(Expr, Vec<Stmt>)>, Option<Vec<Stmt>>),
    For(String, Expr, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    Break,
    Continue,
    Return,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub outs: Vec<String>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub functions: Vec<Rc<Function>>,
    pub body: Vec<Stmt>,
}

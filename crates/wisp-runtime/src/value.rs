//! Runtime values.

use crate::net::NetRef;
use crate::tensor::Tensor;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::rc::Rc;
use wisp_syntax::ast::Expr;

#[derive(Clone)]
pub enum FuncVal {
    Named(String),
    Lambda {
        params: Rc<Vec<String>>,
        body: Rc<Expr>,
        env: Rc<HashMap<String, Value>>,
    },
}

#[derive(Clone)]
pub enum Value {
    Arr(Tensor),
    Str(Rc<str>),
    Cell(Rc<Vec<Value>>),
    Struct(Rc<IndexMap<String, Value>>),
    Func(Rc<FuncVal>),
    Net(NetRef),
    /// A reference to one weight tensor.
    Part(NetRef, Rc<str>),
    /// A reference to a namespace inside a net (`net.L[0]`, `net.L[0].mlp`).
    Path(NetRef, Rc<str>),
    Group(Rc<Vec<(NetRef, String)>>),
    Nothing,
}

impl Value {
    pub fn num(v: f64) -> Value {
        Value::Arr(Tensor::scalar(v))
    }
    pub fn boolean(b: bool) -> Value {
        Value::Arr(Tensor::boolean(b))
    }
    pub fn str(s: impl AsRef<str>) -> Value {
        Value::Str(Rc::from(s.as_ref()))
    }
    pub fn cell(v: Vec<Value>) -> Value {
        Value::Cell(Rc::new(v))
    }
    pub fn strukt(m: IndexMap<String, Value>) -> Value {
        Value::Struct(Rc::new(m))
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Arr(_) => "array",
            Value::Str(_) => "string",
            Value::Cell(_) => "cell",
            Value::Struct(_) => "struct",
            Value::Func(_) => "function",
            Value::Net(_) => "net",
            Value::Part(..) => "part",
            Value::Path(..) => "net path",
            Value::Group(_) => "group",
            Value::Nothing => "nothing",
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

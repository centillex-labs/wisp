//! Tree-walking interpreter.

use crate::builtins;
use crate::harness::Harness;
use crate::index::{self, Idx};
use crate::net::{NetData, NetRef, PartData, Resolved};
use crate::tensor::{Tag, Tensor};
use crate::value::{FuncVal, Value};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::rc::Rc;
use wisp_llvm::jit::NativeScalar;
use wisp_syntax::ast::*;
use wisp_syntax::{Diag, Span};

pub type Env = HashMap<String, Value>;
pub type R<T> = Result<T, Diag>;

enum Flow {
    Normal,
    Break,
    Continue,
    Return,
}

pub fn err(code: &'static str, msg: impl Into<String>, span: Span) -> Diag {
    Diag::new(code, msg, span)
}

pub struct Interp {
    pub funcs: HashMap<String, Rc<Function>>,
    pub harness: Harness,
    pub file: String,
    pub src: Rc<str>,
    pub native: Option<NativeScalar>,
    pub native_calls: u64,
    pub out: String,
    pub echo: bool,
    depth: usize,
    last_stack: Vec<f64>,
    pub timer: Option<std::time::Instant>,
}

/// An assignment path step with its indices already evaluated.
enum EvPost {
    Field(String),
    Index(Vec<Idx>),
    Cell(usize),
}

enum Meta {
    Arr(Vec<usize>, usize),
    Cell(usize),
    Struct,
    Ref(Value),
    Other(&'static str),
    Missing,
}

impl Meta {
    fn kind(&self) -> &'static str {
        match self {
            Meta::Arr(..) => "array",
            Meta::Cell(_) => "cell",
            Meta::Struct => "struct",
            Meta::Ref(v) => v.kind(),
            Meta::Other(k) => k,
            Meta::Missing => "undefined variable",
        }
    }
}

fn meta_of(v: Option<&Value>) -> Meta {
    match v {
        None => Meta::Missing,
        Some(Value::Arr(t)) => Meta::Arr(t.shape.clone(), t.numel()),
        Some(Value::Cell(c)) => Meta::Cell(c.len()),
        Some(Value::Struct(_)) => Meta::Struct,
        Some(r @ (Value::Net(_) | Value::Path(..) | Value::Part(..))) => Meta::Ref(r.clone()),
        Some(other) => Meta::Other(other.kind()),
    }
}

/// Follow already-evaluated steps by reference (no copies).
fn navigate<'a>(root: Option<&'a Value>, evs: &[EvPost]) -> Option<&'a Value> {
    let mut cur = root?;
    for e in evs {
        cur = match (cur, e) {
            (Value::Struct(s), EvPost::Field(f)) => s.get(f)?,
            (Value::Cell(c), EvPost::Cell(i)) => c.get(*i)?,
            _ => return None,
        };
    }
    Some(cur)
}

/// An evaluated index argument.
enum IArg {
    Colon,
    Val(Value),
}

impl Interp {
    pub fn new(file: &str, src: &str) -> Self {
        Interp {
            funcs: HashMap::new(),
            harness: Harness::default(),
            file: file.into(),
            src: Rc::from(src),
            native: None,
            native_calls: 0,
            out: String::new(),
            echo: true,
            depth: 0,
            last_stack: vec![],
            timer: None,
        }
    }

    pub fn print(&mut self, s: &str) {
        if self.echo {
            print!("{s}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        self.out.push_str(s);
    }

    pub fn loc(&self, span: Span) -> String {
        let (l, c, _) = wisp_syntax::diag::locate(&self.src, span.start);
        format!("{}:{l}:{c}", self.file)
    }

    pub fn load_functions(&mut self, prog: &Program) {
        for f in &prog.functions {
            self.funcs.insert(f.name.clone(), f.clone());
        }
    }

    /// Run already-parsed statements against an existing environment.
    /// Used by the kernel, which keeps one environment and budget across calls.
    pub fn eval_program(&mut self, env: &mut Env, prog: &Program) -> R<()> {
        self.load_functions(prog);
        self.block(env, &prog.body)?;
        Ok(())
    }

    /// Run a whole program; returns the final variables.
    pub fn run(&mut self, prog: &Program) -> R<Env> {
        self.load_functions(prog);
        let mut env = Env::new();
        self.block(&mut env, &prog.body)?;
        Ok(env)
    }

    // ------------------------------------------------------------ statements

    fn block(&mut self, env: &mut Env, stmts: &[Stmt]) -> R<Flow> {
        for s in stmts {
            match self.stmt(env, s)? {
                Flow::Normal => {}
                f => return Ok(f),
            }
        }
        Ok(Flow::Normal)
    }

    fn stmt(&mut self, env: &mut Env, s: &Stmt) -> R<Flow> {
        match &s.kind {
            StmtKind::Expr(e) => {
                self.eval_multi(env, e, 0)?;
            }
            StmtKind::Assign(lv, op, rhs) => {
                let v = self.eval(env, rhs)?;
                self.assign(env, lv, v, *op)?;
            }
            StmtKind::Multi(targets, rhs) => {
                let vals = self.eval_multi(env, rhs, targets.len())?;
                if vals.len() < targets.len() {
                    return Err(err(
                        "E-ARGS",
                        format!(
                            "the right side returns {} value(s), {} were requested",
                            vals.len(),
                            targets.len()
                        ),
                        rhs.span,
                    ));
                }
                for (t, v) in targets.iter().zip(vals) {
                    if let Some(lv) = t {
                        self.assign(env, lv, v, AssignOp::Set)?;
                    }
                }
            }
            StmtKind::If(arms, else_) => {
                for (c, body) in arms {
                    let v = self.eval(env, c)?;
                    if self.truthy(&v, c.span)? {
                        return self.block(env, body);
                    }
                }
                if let Some(b) = else_ {
                    return self.block(env, b);
                }
            }
            StmtKind::For(var, iter, body) => {
                let it = self.eval(env, iter)?;
                let items: Vec<Value> = match it {
                    Value::Arr(t) => {
                        if t.is_empty() {
                            vec![]
                        } else {
                            let (r, c) = (t.rows(), t.cols());
                            (0..c)
                                .map(|j| {
                                    let d: Vec<f64> = (0..r).map(|i| t.data[i * c + j]).collect();
                                    let mut col = Tensor::new(vec![r, 1], d, t.tag);
                                    col.logical = t.logical;
                                    Value::Arr(col)
                                })
                                .collect()
                        }
                    }
                    Value::Cell(c) => c.iter().cloned().collect(),
                    Value::Group(g) => g
                        .iter()
                        .map(|(n, p)| Value::Part(n.clone(), Rc::from(p.as_str())))
                        .collect(),
                    other => {
                        return Err(err(
                            "E-TYPE",
                            format!("cannot loop over a {}", other.kind()),
                            iter.span,
                        ))
                    }
                };
                for item in items {
                    env.insert(var.clone(), item);
                    match self.block(env, body)? {
                        Flow::Break => break,
                        Flow::Return => return Ok(Flow::Return),
                        _ => {}
                    }
                }
            }
            StmtKind::While(c, body) => loop {
                let v = self.eval(env, c)?;
                if !self.truthy(&v, c.span)? {
                    break;
                }
                match self.block(env, body)? {
                    Flow::Break => break,
                    Flow::Return => return Ok(Flow::Return),
                    _ => {}
                }
            },
            StmtKind::Break => return Ok(Flow::Break),
            StmtKind::Continue => return Ok(Flow::Continue),
            StmtKind::Return => return Ok(Flow::Return),
        }
        Ok(Flow::Normal)
    }

    pub fn truthy(&mut self, v: &Value, span: Span) -> R<bool> {
        match v {
            Value::Arr(t) => Ok(!t.is_empty() && t.data.iter().all(|&x| x != 0.0)),
            other => Err(err(
                "E-TYPE",
                format!("a condition must be a number, got a {}", other.kind()),
                span,
            )),
        }
    }

    // ------------------------------------------------------------ assignment

    /// Combine a resolved target with the RHS without evaluating its path again.
    fn assignment_value(
        &mut self,
        cur: Option<Value>,
        v: Value,
        op: AssignOp,
        span: Span,
    ) -> R<Value> {
        if op == AssignOp::Set {
            return Ok(v);
        }
        match (cur, op) {
            (None, AssignOp::Add) if matches!(v, Value::Cell(_)) => Ok(v),
            (Some(Value::Cell(c)), AssignOp::Add) => {
                let mut c = (*c).clone();
                match v {
                    Value::Cell(more) => c.extend(more.iter().cloned()),
                    other => c.push(other),
                }
                Ok(Value::cell(c))
            }
            (Some(cur), op) => {
                let bop = match op {
                    AssignOp::Add => BinOp::Add,
                    AssignOp::Sub => BinOp::Sub,
                    AssignOp::Mul => BinOp::MatMul,
                    _ => BinOp::Div,
                };
                self.binop(bop, cur, v, span)
            }
            (None, _) => Err(err(
                "E-UNDEFINED",
                "compound assignment target is not defined",
                span,
            )),
        }
    }

    /// Assignment. Index arguments are evaluated first, while the variable is still
    /// visible (so `x[numel(x)] = v` works); then the value is updated in place.
    fn assign(&mut self, env: &mut Env, lv: &LValue, v: Value, op: AssignOp) -> R<()> {
        let span = lv.span;
        let mut evs: Vec<EvPost> = Vec::new();
        for (k, p) in lv.path.iter().enumerate() {
            let meta = meta_of(navigate(env.get(&lv.base), &evs));
            if let Meta::Ref(r) = meta {
                // a net, path or part: write through the reference; the container is unchanged
                return self.assign_ref(env, r, &lv.path[k..], v, op, span);
            }
            match p {
                Post::Field(_) | Post::DynField(_) => {
                    if !matches!(meta, Meta::Struct | Meta::Missing) {
                        return Err(err(
                            "E-TYPE",
                            format!("cannot set a field on a {}", meta.kind()),
                            span,
                        ));
                    }
                    let name = self.field_name(env, p, span)?;
                    evs.push(EvPost::Field(name));
                }
                Post::Index(args) => {
                    if k + 1 != lv.path.len() {
                        return Err(err(
                            "E-TYPE",
                            "cannot index further into an array element",
                            span,
                        ));
                    }
                    let (shape, numel) = match meta {
                        Meta::Arr(s, n) => (s, n),
                        Meta::Missing => (vec![0, 0], 0),
                        other => {
                            return Err(err(
                                "E-TYPE",
                                format!("cannot index-assign into a {}", other.kind()),
                                span,
                            ))
                        }
                    };
                    let idxs = self.index_list(env, args, &shape, numel, true, false, span)?;
                    evs.push(EvPost::Index(idxs));
                }
                Post::CellIndex(args) => {
                    let len = match meta {
                        Meta::Cell(n) => n,
                        Meta::Missing => 0,
                        other => {
                            return Err(err(
                                "E-TYPE",
                                format!("{{ }} assigns into cells, not a {}", other.kind()),
                                span,
                            ))
                        }
                    };
                    if args.len() != 1 {
                        return Err(err("E-INDEX", "cell indexing takes one position", span));
                    }
                    self.last_stack.push(len as f64 - 1.0);
                    let iv = self.eval(env, &args[0].value);
                    self.last_stack.pop();
                    let i = self.scalar_index(iv?, span)?;
                    evs.push(EvPost::Cell(i));
                }
                Post::Call(_) | Post::Transpose => unreachable!(),
            }
        }
        let root = env.remove(&lv.base);
        let new = self.apply_evs(root, &evs, v, op, span)?;
        env.insert(lv.base.clone(), new);
        Ok(())
    }

    fn apply_evs(
        &mut self,
        cur: Option<Value>,
        evs: &[EvPost],
        v: Value,
        op: AssignOp,
        span: Span,
    ) -> R<Value> {
        let Some(first) = evs.first() else {
            return self.assignment_value(cur, v, op, span);
        };
        match first {
            EvPost::Field(name) => {
                let mut m = match cur {
                    Some(Value::Struct(s)) => Rc::try_unwrap(s).unwrap_or_else(|s| (*s).clone()),
                    _ => IndexMap::new(),
                };
                let child = m
                    .get_mut(name)
                    .map(|x| std::mem::replace(x, Value::Nothing));
                let nv = self.apply_evs(child, &evs[1..], v, op, span)?;
                m.insert(name.clone(), nv);
                Ok(Value::Struct(Rc::new(m)))
            }
            EvPost::Cell(i) => {
                let mut c: Vec<Value> = match cur {
                    Some(Value::Cell(c)) => Rc::try_unwrap(c).unwrap_or_else(|c| (*c).clone()),
                    _ => vec![],
                };
                while c.len() <= *i {
                    c.push(Value::Arr(Tensor::empty()));
                }
                let child = std::mem::replace(&mut c[*i], Value::Nothing);
                let nv = self.apply_evs(Some(child), &evs[1..], v, op, span)?;
                c[*i] = nv;
                Ok(Value::Cell(Rc::new(c)))
            }
            EvPost::Index(idxs) => {
                let mut t = match cur {
                    Some(Value::Arr(t)) => t,
                    _ => Tensor::empty(),
                };
                let old = if op == AssignOp::Set {
                    None
                } else {
                    Some(Value::Arr(
                        index::read(&t, idxs).map_err(|m| err("E-INDEX", m, span))?,
                    ))
                };
                let v = self.assignment_value(old, v, op, span)?;
                let val = self.to_arr(v, span)?;
                index::grow_for(&mut t, idxs).map_err(|m| err("E-SHAPE", m, span))?;
                index::write(&mut t, idxs, &val).map_err(|m| err("E-SHAPE", m, span))?;
                t.logical = val.logical && t.data.iter().all(|&x| x == 0.0 || x == 1.0);
                Ok(Value::Arr(t))
            }
        }
    }

    fn field_name(&mut self, env: &mut Env, p: &Post, span: Span) -> R<String> {
        match p {
            Post::Field(f) => Ok(f.clone()),
            Post::DynField(e) => match self.eval(env, e)? {
                Value::Str(s) => Ok(s.to_string()),
                _ => Err(err("E-TYPE", "a dynamic field name must be a string", span)),
            },
            _ => unreachable!(),
        }
    }

    /// Write through a reference (net, path or part).
    fn assign_ref(
        &mut self,
        env: &mut Env,
        target: Value,
        path: &[Post],
        v: Value,
        op: AssignOp,
        span: Span,
    ) -> R<()> {
        let mut cur = target;
        let mut k = 0;
        while k < path.len() {
            let last_op = k == path.len() - 1;
            match (&cur, &path[k]) {
                (Value::Net(n), Post::Field(_) | Post::DynField(_))
                | (Value::Path(n, _), Post::Field(_) | Post::DynField(_)) => {
                    let n = n.clone();
                    let name = self.field_name(env, &path[k], span)?;
                    let full = match &cur {
                        Value::Path(_, p) => format!("{p}.{name}"),
                        _ => name.clone(),
                    };
                    let res = n.borrow().resolve(&full);
                    match res {
                        Resolved::Part => cur = Value::Part(n, Rc::from(full.as_str())),
                        Resolved::Prefix => cur = Value::Path(n, Rc::from(full.as_str())),
                        Resolved::Meta(old) if matches!(cur, Value::Net(_)) => {
                            if !last_op {
                                return Err(err(
                                    "E-TYPE",
                                    format!("assign the whole metadata field `{name}`"),
                                    span,
                                ));
                            }
                            let v = self.assignment_value(Some(old), v, op, span)?;
                            n.borrow_mut().meta.insert(name, v);
                            return Ok(());
                        }
                        Resolved::Missing if matches!(cur, Value::Net(_)) && last_op => {
                            let v = self.assignment_value(None, v, op, span)?;
                            if let Value::Arr(_) = v {
                                return Err(err(
                                    "E-UNDEFINED",
                                    format!("the net has no part `{full}`"),
                                    span,
                                )
                                .hint("parts cannot be created by assignment; metadata fields hold strings, structs or cells"));
                            }
                            n.borrow_mut().meta.insert(name, v);
                            return Ok(());
                        }
                        _ => return Err(self.no_part(&n, &full, span)),
                    }
                    k += 1;
                }
                (Value::Path(n, p), Post::Index(args)) => {
                    let n = n.clone();
                    let p = p.to_string();
                    let full = self.path_index(env, &n, &p, args, span)?;
                    cur = full;
                    k += 1;
                }
                (Value::Part(n, p), Post::Index(args)) if last_op => {
                    let n = n.clone();
                    let p = p.to_string();
                    let (shape, numel) = {
                        let nb = n.borrow();
                        let pd = &nb.parts[&p];
                        (pd.tensor().shape, pd.data.len())
                    };
                    let idxs = self.index_list(env, args, &shape, numel, false, true, span)?;
                    let old = if op == AssignOp::Set {
                        None
                    } else {
                        let t = n.borrow().parts[&p].tensor();
                        Some(Value::Arr(
                            index::read(&t, &idxs).map_err(|m| err("E-INDEX", m, span))?,
                        ))
                    };
                    let v = self.assignment_value(old, v, op, span)?;
                    let val = self.to_arr(v, span)?;
                    let mut nb = n.borrow_mut();
                    nb.native = None;
                    let pd = nb.parts.get_mut(&p).unwrap();
                    write_part(pd, &shape, &idxs, &val).map_err(|m| err("E-SHAPE", m, span))?;
                    return Ok(());
                }
                _ => {
                    return Err(err(
                        "E-TYPE",
                        format!("cannot assign into a {} this way", cur.kind()),
                        span,
                    ))
                }
            }
        }
        // whole-part assignment
        match cur {
            Value::Part(n, p) => {
                let old = if op == AssignOp::Set {
                    None
                } else {
                    Some(Value::Arr(n.borrow().parts[p.as_ref()].tensor()))
                };
                let v = self.assignment_value(old, v, op, span)?;
                let val = self.to_arr(v, span)?;
                let mut nb = n.borrow_mut();
                nb.native = None;
                let pd = nb.parts.get_mut(p.as_ref()).unwrap();
                let shape = pd.tensor().shape;
                let all: Vec<Idx> = shape.iter().map(|&d| Idx::All(d)).collect();
                write_part(pd, &shape, &all, &val)
                    .map_err(|m| err("E-SHAPE", format!("{m} (part `{p}`)"), span))?;
                Ok(())
            }
            other => Err(err(
                "E-TYPE",
                format!("cannot assign an array to a {}", other.kind()),
                span,
            )),
        }
    }

    fn no_part(&self, n: &NetRef, full: &str, span: Span) -> Diag {
        let nb = n.borrow();
        let mut tops: Vec<String> = nb
            .parts
            .keys()
            .map(|k| k.split(['.', '[']).next().unwrap_or(k).to_string())
            .collect();
        tops.dedup();
        tops.extend(nb.meta.keys().cloned());
        let sugg = closest(full, nb.parts.keys().map(|s| s.as_str()));
        let mut d = err("E-UNDEFINED", format!("the net has no part `{full}`"), span);
        d = match sugg {
            Some(s) => d.hint(format!("did you mean `{s}`?")),
            None => d.hint(format!("top-level names: {}", tops.join(", "))),
        };
        d
    }

    fn path_index(
        &mut self,
        env: &mut Env,
        n: &NetRef,
        p: &str,
        args: &[Arg],
        span: Span,
    ) -> R<Value> {
        if args.len() != 1 {
            return Err(err(
                "E-INDEX",
                format!("`{p}[...]` takes one position"),
                span,
            ));
        }
        let subs = n.borrow().sub_indices(p);
        self.last_stack
            .push(subs.last().map(|&x| x as f64).unwrap_or(-1.0));
        let v = self.eval(env, &args[0].value);
        self.last_stack.pop();
        let i = self.scalar_index(v?, span)?;
        let full = format!("{p}[{i}]");
        let res = n.borrow().resolve(&full);
        match res {
            Resolved::Part => Ok(Value::Part(n.clone(), Rc::from(full.as_str()))),
            Resolved::Prefix => Ok(Value::Path(n.clone(), Rc::from(full.as_str()))),
            _ => Err(err(
                "E-INDEX-RANGE",
                format!(
                    "`{full}` does not exist{}",
                    if subs.is_empty() {
                        String::new()
                    } else {
                        format!(" (valid: {p}[0..{}])", subs.last().unwrap())
                    }
                ),
                span,
            )),
        }
    }

    fn scalar_index(&mut self, v: Value, span: Span) -> R<usize> {
        let t = self.to_arr(v, span)?;
        match t.scalar_value() {
            Some(x) if x >= 0.0 && x == x.trunc() => Ok(x as usize),
            _ => Err(err(
                "E-INDEX",
                "expected one non-negative integer position (indices start at 0)",
                span,
            )),
        }
    }

    /// Evaluate index arguments for a container of `shape`.
    #[allow(clippy::too_many_arguments)]
    fn index_list(
        &mut self,
        env: &mut Env,
        args: &[Arg],
        shape: &[usize],
        numel: usize,
        grow: bool,
        named_ok: bool,
        span: Span,
    ) -> R<Vec<Idx>> {
        // named arguments: out= -> dim 0, in= -> dim 1
        let named = args.iter().any(|a| a.name.is_some());
        let ordered: Vec<Option<&Arg>> = if named {
            if !named_ok && shape.len() != 2 {
                return Err(err("E-INDEX", "named indices need a 2-D weight", span));
            }
            let mut slots: Vec<Option<&Arg>> = vec![None, None];
            for (k, a) in args.iter().enumerate() {
                let d = match a.name.as_deref() {
                    Some("out") => 0,
                    Some("in") => 1,
                    Some(other) => {
                        return Err(err(
                            "E-INDEX",
                            format!("unknown index name `{other}` (use out= or in=)"),
                            a.value.span,
                        ))
                    }
                    None => k,
                };
                if d > 1 || slots[d].is_some() {
                    return Err(err(
                        "E-INDEX",
                        "each dimension can be indexed once",
                        a.value.span,
                    ));
                }
                slots[d] = Some(a);
            }
            slots
        } else {
            args.iter().map(Some).collect()
        };
        let k = ordered.len();
        let mut out = Vec::with_capacity(k);
        for (d, a) in ordered.iter().enumerate() {
            let dim = if k == 1 {
                numel
            } else {
                shape.get(d).copied().unwrap_or(1)
            };
            let ia = match a {
                None => IArg::Colon,
                Some(a) => match a.value.kind {
                    ExprKind::Colon => IArg::Colon,
                    _ => {
                        self.last_stack.push(dim as f64 - 1.0);
                        let v = self.eval(env, &a.value);
                        self.last_stack.pop();
                        IArg::Val(v?)
                    }
                },
            };
            let sp = a.map(|a| a.value.span).unwrap_or(span);
            out.push(match ia {
                IArg::Colon => Idx::All(dim),
                IArg::Val(v) => {
                    let t = self.to_arr(v, sp)?;
                    Idx::from_tensor(&t, dim, grow).map_err(|m| {
                        let mut m = m;
                        if k > 1 {
                            m = format!("{m} in dimension {d}, which has size {dim}");
                        }
                        err("E-INDEX-RANGE", m, sp)
                    })?
                }
            });
        }
        Ok(out)
    }

    // ------------------------------------------------------------ expressions

    pub fn eval(&mut self, env: &mut Env, e: &Expr) -> R<Value> {
        let mut v = self.eval_multi(env, e, 1)?;
        if v.is_empty() {
            return Err(err("E-ARGS", "this call returns no value", e.span));
        }
        Ok(v.swap_remove(0))
    }

    pub fn eval_multi(&mut self, env: &mut Env, e: &Expr, nargout: usize) -> R<Vec<Value>> {
        Ok(match &e.kind {
            ExprKind::Num(v) => vec![Value::num(*v)],
            ExprKind::Str(s) => vec![Value::str(s)],
            ExprKind::Bool(b) => vec![Value::boolean(*b)],
            ExprKind::Last => match self.last_stack.last() {
                Some(&l) => vec![Value::num(l)],
                None => return Err(err("E-PARSE", "`last` is only valid inside [ ]", e.span)),
            },
            ExprKind::Colon => return Err(err("E-PARSE", "`:` is only valid inside [ ]", e.span)),
            ExprKind::Tilde => {
                return Err(err(
                    "E-PARSE",
                    "`~` is only valid as an output placeholder",
                    e.span,
                ))
            }
            ExprKind::Ident(n) => {
                if let Some(v) = env.get(n) {
                    vec![v.clone()]
                } else if self.funcs.contains_key(n) || builtins::exists(n) {
                    self.call_named(n, vec![], nargout, e.span)?
                } else {
                    let sugg = closest(
                        n,
                        env.keys()
                            .map(|s| s.as_str())
                            .chain(self.funcs.keys().map(|s| s.as_str()))
                            .chain(builtins::NAMES.iter().copied()),
                    );
                    let mut d = err("E-UNDEFINED", format!("`{n}` is not defined"), e.span);
                    if let Some(s) = sugg {
                        d = d.hint(format!("did you mean `{s}`?"));
                    }
                    return Err(d);
                }
            }
            ExprKind::Matrix(rows) => vec![self.matrix(env, rows, e.span)?],
            ExprKind::Cell(rows) => {
                let mut v = Vec::new();
                for r in rows {
                    for x in r {
                        v.push(self.eval(env, x)?);
                    }
                }
                vec![Value::cell(v)]
            }
            ExprKind::Lambda(params, body) => vec![Value::Func(Rc::new(FuncVal::Lambda {
                params: params.clone(),
                body: body.clone(),
                env: Rc::new(env.clone()),
            }))],
            ExprKind::FuncRef(n) => {
                if !self.funcs.contains_key(n) && !builtins::exists(n) {
                    return Err(err(
                        "E-UNDEFINED",
                        format!("no function named `{n}`"),
                        e.span,
                    ));
                }
                vec![Value::Func(Rc::new(FuncVal::Named(n.clone())))]
            }
            ExprKind::Unary(op, a) => {
                let v = self.eval(env, a)?;
                vec![self.unop(*op, v, e.span)?]
            }
            ExprKind::Binary(BinOp::And, a, b) | ExprKind::Binary(BinOp::Or, a, b) => {
                let is_and = matches!(e.kind, ExprKind::Binary(BinOp::And, ..));
                let av = self.eval(env, a)?;
                let at = self.scalar_bool(&av, a.span)?;
                if is_and && !at {
                    return Ok(vec![Value::boolean(false)]);
                }
                if !is_and && at {
                    return Ok(vec![Value::boolean(true)]);
                }
                let bv = self.eval(env, b)?;
                vec![Value::boolean(self.scalar_bool(&bv, b.span)?)]
            }
            ExprKind::Binary(op, a, b) => {
                let av = self.eval(env, a)?;
                let bv = self.eval(env, b)?;
                vec![self.binop(*op, av, bv, e.span)?]
            }
            ExprKind::Range(a, b, s) => {
                let av = self.eval(env, a)?;
                let bv = self.eval(env, b)?;
                let sv = match s {
                    Some(s) => Some(self.eval(env, s)?),
                    None => None,
                };
                vec![self.range(av, bv, sv, e.span)?]
            }
            ExprKind::Postfix(base, post) => {
                // function call by name
                if let (ExprKind::Ident(n), Post::Call(args)) = (&base.kind, post) {
                    if !env.contains_key(n) {
                        if self.funcs.contains_key(n) || builtins::exists(n) {
                            let argv = self.eval_args(env, args)?;
                            return self.call_named(n, argv, nargout, e.span);
                        }
                        let sugg = closest(
                            n,
                            self.funcs
                                .keys()
                                .map(|s| s.as_str())
                                .chain(builtins::NAMES.iter().copied()),
                        );
                        let mut d = err(
                            "E-UNDEFINED",
                            format!("no function or variable named `{n}`"),
                            base.span,
                        );
                        if let Some(s) = sugg {
                            d = d.hint(format!("did you mean `{s}`?"));
                        }
                        return Err(d);
                    }
                }
                let bv = self.eval(env, base)?;
                if let Post::Call(args) = post {
                    return match bv {
                        Value::Func(f) => {
                            let argv = self.eval_args(env, args)?;
                            self.call_value(&f, argv, nargout, e.span)
                        }
                        Value::Arr(_) | Value::Part(..) | Value::Cell(_) | Value::Str(_) => {
                            Err(err(
                                "E-PAREN-INDEX",
                                "parentheses call functions; this is an array",
                                e.span,
                            )
                            .hint("to index, write x[...] with 0-based positions"))
                        }
                        other => Err(err(
                            "E-TYPE",
                            format!("a {} cannot be called", other.kind()),
                            e.span,
                        )),
                    };
                }
                vec![self.apply_post(env, bv, post, e.span)?]
            }
        })
    }

    fn eval_args(&mut self, env: &mut Env, args: &[Arg]) -> R<Vec<Value>> {
        let mut v = Vec::with_capacity(args.len());
        for a in args {
            if let ExprKind::Colon = a.value.kind {
                return Err(err("E-PARSE", "`:` is only valid inside [ ]", a.value.span));
            }
            v.push(self.eval(env, &a.value)?);
        }
        Ok(v)
    }

    fn scalar_bool(&mut self, v: &Value, span: Span) -> R<bool> {
        match v {
            Value::Arr(t) if t.is_scalar() => Ok(t.data[0] != 0.0),
            _ => Err(err("E-TYPE", "&& and || need scalar operands", span)),
        }
    }

    pub fn apply_post(&mut self, env: &mut Env, v: Value, post: &Post, span: Span) -> R<Value> {
        match post {
            Post::Field(_) | Post::DynField(_) => {
                let name = self.field_name(env, post, span)?;
                self.field(v, &name, span)
            }
            Post::Transpose => {
                let t = self.to_arr(v, span)?;
                Ok(Value::Arr(
                    t.transpose().map_err(|m| err("E-SHAPE", m, span))?,
                ))
            }
            Post::Index(args) => self.index_value(env, v, args, span),
            Post::CellIndex(args) => match v {
                Value::Cell(c) => {
                    if args.len() != 1 {
                        return Err(err("E-INDEX", "cell indexing takes one position", span));
                    }
                    self.last_stack.push(c.len() as f64 - 1.0);
                    let iv = self.eval(env, &args[0].value);
                    self.last_stack.pop();
                    let i = self.scalar_index(iv?, span)?;
                    c.get(i).cloned().ok_or_else(|| {
                        err(
                            "E-INDEX-RANGE",
                            if c.is_empty() {
                                format!("index {i} is out of range: the cell is empty")
                            } else {
                                format!(
                                    "index {i} is out of range for a cell of {} (valid: 0..{})",
                                    c.len(),
                                    c.len() - 1
                                )
                            },
                            span,
                        )
                    })
                }
                other => Err(err(
                    "E-TYPE",
                    format!("{{ }} indexes cells, not a {}", other.kind()),
                    span,
                )),
            },
            Post::Call(_) => unreachable!(),
        }
    }

    pub fn field(&mut self, v: Value, name: &str, span: Span) -> R<Value> {
        match v {
            Value::Struct(s) => s.get(name).cloned().ok_or_else(|| {
                let sugg = closest(name, s.keys().map(|k| k.as_str()));
                let d = err("E-UNDEFINED", format!("no field `{name}`"), span);
                match sugg {
                    Some(x) => d.hint(format!("did you mean `{x}`?")),
                    None => d.hint(format!(
                        "fields: {}",
                        s.keys().cloned().collect::<Vec<_>>().join(", ")
                    )),
                }
            }),
            Value::Net(ref n) | Value::Path(ref n, _) => {
                let n = n.clone();
                let full = match &v {
                    Value::Path(_, p) => format!("{p}.{name}"),
                    _ => name.to_string(),
                };
                let res = n.borrow().resolve(&full);
                match res {
                    Resolved::Part => Ok(Value::Part(n.clone(), Rc::from(full.as_str()))),
                    Resolved::Prefix => Ok(Value::Path(n.clone(), Rc::from(full.as_str()))),
                    Resolved::Meta(m) if matches!(v, Value::Net(_)) => Ok(m),
                    _ => Err(self.no_part(&n, &full, span)),
                }
            }
            other => Err(err(
                "E-TYPE",
                format!("a {} has no fields", other.kind()),
                span,
            )),
        }
    }

    fn index_value(&mut self, env: &mut Env, v: Value, args: &[Arg], span: Span) -> R<Value> {
        match v {
            Value::Arr(t) => {
                let idxs =
                    self.index_list(env, args, &t.shape.clone(), t.numel(), false, false, span)?;
                Ok(Value::Arr(
                    index::read(&t, &idxs).map_err(|m| err("E-INDEX", m, span))?,
                ))
            }
            Value::Part(n, p) => {
                let t = n.borrow().parts[p.as_ref()].tensor();
                let idxs =
                    self.index_list(env, args, &t.shape.clone(), t.numel(), false, true, span)?;
                Ok(Value::Arr(
                    index::read(&t, &idxs).map_err(|m| err("E-INDEX", m, span))?,
                ))
            }
            Value::Path(n, p) => self.path_index(env, &n, &p, args, span),
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let idxs = self.index_list(
                    env,
                    args,
                    &[1, chars.len()],
                    chars.len(),
                    false,
                    false,
                    span,
                )?;
                let pos = index::flat_positions(&idxs, &[1, chars.len()])
                    .map_err(|m| err("E-INDEX", m, span))?;
                Ok(Value::str(
                    pos.iter().map(|&i| chars[i]).collect::<String>(),
                ))
            }
            Value::Cell(c) => {
                let idxs =
                    self.index_list(env, args, &[1, c.len()], c.len(), false, false, span)?;
                let pos = index::flat_positions(&idxs, &[1, c.len()])
                    .map_err(|m| err("E-INDEX", m, span))?;
                Ok(Value::cell(pos.iter().map(|&i| c[i].clone()).collect()))
            }
            Value::Group(g) => {
                let idxs =
                    self.index_list(env, args, &[1, g.len()], g.len(), false, false, span)?;
                let pos = index::flat_positions(&idxs, &[1, g.len()])
                    .map_err(|m| err("E-INDEX", m, span))?;
                if pos.len() == 1 {
                    let (n, p) = &g[pos[0]];
                    return Ok(Value::Part(n.clone(), Rc::from(p.as_str())));
                }
                Ok(Value::Group(Rc::new(
                    pos.iter().map(|&i| g[i].clone()).collect(),
                )))
            }
            other => Err(err(
                "E-TYPE",
                format!("a {} cannot be indexed", other.kind()),
                span,
            )),
        }
    }

    fn matrix(&mut self, env: &mut Env, rows: &[Vec<Expr>], span: Span) -> R<Value> {
        let mut vals: Vec<Vec<Value>> = Vec::new();
        for r in rows {
            let mut row = Vec::new();
            for x in r {
                row.push(self.eval(env, x)?);
            }
            vals.push(row);
        }
        let flat: Vec<&Value> = vals.iter().flatten().collect();
        if flat.is_empty() {
            return Ok(Value::Arr(Tensor::empty()));
        }
        if flat.iter().all(|v| matches!(v, Value::Str(_))) {
            let s: String = flat
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            return Ok(Value::str(s));
        }
        if flat.iter().any(|v| matches!(v, Value::Cell(_))) {
            let mut out = Vec::new();
            for v in flat {
                match v {
                    Value::Cell(c) => out.extend(c.iter().cloned()),
                    other => out.push(other.clone()),
                }
            }
            return Ok(Value::cell(out));
        }
        // numeric: concatenate horizontally within rows, then vertically
        let mut row_ts: Vec<Tensor> = Vec::new();
        for r in vals {
            let mut parts: Vec<Tensor> = Vec::new();
            for v in r {
                let t = self.to_arr(v, span)?;
                if t.is_empty() {
                    continue;
                }
                parts.push(t);
            }
            if parts.is_empty() {
                continue;
            }
            let h = parts[0].rows();
            let mut tag = Tag::Lit;
            let mut logical = true;
            for p in &parts {
                if p.shape.len() != 2 || p.rows() != h {
                    return Err(err(
                        "E-SHAPE",
                        format!(
                            "cannot place a {} next to a {}-row element",
                            p.shape_str(),
                            h
                        ),
                        span,
                    ));
                }
                tag = tag.max(p.tag);
                logical &= p.logical;
            }
            let w: usize = parts.iter().map(|p| p.cols()).sum();
            let mut d = vec![0.0; h * w];
            let mut off = 0;
            for p in &parts {
                let c = p.cols();
                for i in 0..h {
                    d[i * w + off..i * w + off + c].copy_from_slice(&p.data[i * c..(i + 1) * c]);
                }
                off += c;
            }
            let mut t = Tensor::new(vec![h, w], d, tag);
            t.logical = logical;
            row_ts.push(t);
        }
        if row_ts.is_empty() {
            return Ok(Value::Arr(Tensor::empty()));
        }
        if row_ts.len() == 1 {
            return Ok(Value::Arr(row_ts.pop().unwrap()));
        }
        let w = row_ts[0].cols();
        let mut d = Vec::new();
        let mut h = 0;
        let mut tag = Tag::Lit;
        for t in &row_ts {
            if t.cols() != w {
                return Err(err(
                    "E-SHAPE",
                    format!("rows have different widths ({} and {})", w, t.cols()),
                    span,
                ));
            }
            d.extend_from_slice(&t.data);
            h += t.rows();
            tag = tag.max(t.tag);
        }
        Ok(Value::Arr(Tensor::new(vec![h, w], d, tag)))
    }

    fn range(&mut self, a: Value, b: Value, s: Option<Value>, span: Span) -> R<Value> {
        let a = self.to_arr(a, span)?;
        let b = self.to_arr(b, span)?;
        let s = match s {
            Some(s) => self.to_arr(s, span)?,
            None => Tensor::scalar(1.0),
        };
        let (av, bv, sv) = match (a.scalar_value(), b.scalar_value(), s.scalar_value()) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => return Err(err("E-SHAPE", "range ends and step must be scalars", span)),
        };
        if sv == 0.0 {
            return Err(err("E-SHAPE", "a range step cannot be 0", span));
        }
        let n = ((bv - av) / sv + 1e-9).floor() + 1.0;
        let n = if n > 0.0 { n as usize } else { 0 };
        let d: Vec<f64> = (0..n).map(|k| av + k as f64 * sv).collect();
        let tag = a.tag.max(b.tag).max(s.tag);
        Ok(Value::Arr(Tensor::new(vec![1, n], d, Tag::derived(tag))))
    }

    pub fn to_arr(&self, v: Value, span: Span) -> R<Tensor> {
        match v {
            Value::Arr(t) => Ok(t),
            Value::Part(n, p) => Ok(n.borrow().parts[p.as_ref()].tensor()),
            other => Err(err(
                "E-TYPE",
                format!("expected an array, got a {}", other.kind()),
                span,
            )),
        }
    }

    fn unop(&mut self, op: UnOp, v: Value, span: Span) -> R<Value> {
        let t = self.to_arr(v, span)?;
        Ok(Value::Arr(match op {
            UnOp::Neg => t.map(|x| -x),
            UnOp::Plus => t,
            UnOp::Not => {
                let mut r = t.map(|x| if x == 0.0 { 1.0 } else { 0.0 });
                r.logical = true;
                r
            }
        }))
    }

    pub fn binop(&mut self, op: BinOp, a: Value, b: Value, span: Span) -> R<Value> {
        // Native decoder weights are immutable; edits are installed via bound readout bundles.
        if crate::decoder_api::is_decoder(Some(&a)) || crate::decoder_api::is_decoder(Some(&b)) {
            return Err(err(
                "E-UNSUPPORTED",
                "use token_head and with_head to edit native models",
                span,
            ));
        }
        // net arithmetic
        match (&a, &b) {
            (Value::Net(x), Value::Net(y)) if matches!(op, BinOp::Add | BinOp::Sub) => {
                return Ok(Value::Net(
                    net_combine(x, y, op == BinOp::Sub)
                        .map_err(|m| err("E-SHAPE", m, span))?
                        .into_ref(),
                ));
            }
            (Value::Arr(k), Value::Net(n)) | (Value::Net(n), Value::Arr(k))
                if matches!(op, BinOp::MatMul | BinOp::ElMul) && k.is_scalar() =>
            {
                return Ok(Value::Net(net_scale(n, k.data[0], k.tag).into_ref()));
            }
            (Value::Net(n), Value::Arr(k))
                if matches!(op, BinOp::Div | BinOp::ElDiv) && k.is_scalar() =>
            {
                return Ok(Value::Net(net_scale(n, 1.0 / k.data[0], k.tag).into_ref()));
            }
            (Value::Str(x), Value::Str(y)) if matches!(op, BinOp::Eq | BinOp::Ne) => {
                return Ok(Value::boolean((x == y) == (op == BinOp::Eq)));
            }
            (Value::Net(_), _) | (_, Value::Net(_)) => {
                return Err(err(
                    "E-TYPE",
                    "nets support +, - and scalar * or / only",
                    span,
                ));
            }
            _ => {}
        }
        let x = self.to_arr(a, span)?;
        let y = self.to_arr(b, span)?;
        let sh = |m: String| err("E-SHAPE", m, span);
        let cmp = |f: fn(f64, f64) -> bool| -> R<Value> {
            let mut t = x
                .zip(&y, |p, q| if f(p, q) { 1.0 } else { 0.0 })
                .map_err(sh)?;
            t.logical = true;
            Ok(Value::Arr(t))
        };
        Ok(Value::Arr(match op {
            BinOp::Add => x.zip(&y, |p, q| p + q).map_err(sh)?,
            BinOp::Sub => x.zip(&y, |p, q| p - q).map_err(sh)?,
            BinOp::ElMul => x.zip(&y, |p, q| p * q).map_err(sh)?,
            BinOp::ElDiv => x.zip(&y, |p, q| p / q).map_err(sh)?,
            BinOp::ElPow => x.zip(&y, f64::powf).map_err(sh)?,
            BinOp::Pow => {
                if !y.is_scalar() || !x.is_scalar() {
                    return Err(err(
                        "E-SHAPE",
                        "^ needs scalars; use .^ for elementwise powers",
                        span,
                    ));
                }
                x.zip(&y, f64::powf).map_err(sh)?
            }
            BinOp::Div => {
                if !y.is_scalar() {
                    return Err(err("E-SHAPE", "A / B needs a scalar B; use ./ for elementwise division or A \\ B to solve", span));
                }
                x.zip(&y, |p, q| p / q).map_err(sh)?
            }
            BinOp::MatMul => {
                if x.is_scalar() || y.is_scalar() {
                    x.zip(&y, |p, q| p * q).map_err(sh)?
                } else {
                    let probe = self.probe_check(&x, &y, "*", span)?;
                    let r = x.matmul(&y).map_err(sh)?;
                    if probe {
                        r.with_tag(Tag::Act)
                    } else {
                        r
                    }
                }
            }
            BinOp::LeftDiv => {
                let probe = self.probe_check(&x, &y, "\\", span)?;
                let r = x.left_divide(&y).map_err(sh)?;
                if probe {
                    r.with_tag(Tag::Act)
                } else {
                    r
                }
            }
            BinOp::Eq => return cmp(|p, q| p == q),
            BinOp::Ne => return cmp(|p, q| p != q),
            BinOp::Lt => return cmp(|p, q| p < q),
            BinOp::Le => return cmp(|p, q| p <= q),
            BinOp::Gt => return cmp(|p, q| p > q),
            BinOp::Ge => return cmp(|p, q| p >= q),
            BinOp::And | BinOp::Or => unreachable!(),
        }))
    }

    /// The probe rule: weights times a non-scalar literal is a hand-built forward pass.
    fn probe_check(&mut self, x: &Tensor, y: &Tensor, _op: &str, span: Span) -> R<bool> {
        let weighty = |t: Tag| t == Tag::W || t == Tag::Alg;
        let probe = (weighty(x.tag) && y.tag == Tag::Lit && !y.is_scalar())
            || (weighty(y.tag) && x.tag == Tag::Lit && !x.is_scalar());
        if probe {
            let loc = self.loc(span);
            let literal = if x.tag == Tag::Lit { x } else { y };
            let data = self
                .harness
                .resolve_data(&Value::Arr(literal.clone()))
                .map_err(|(c, m)| err(c, m, span))?;
            self.harness.observe("probe", 1, &loc, &data).map_err(|m| {
                err("E-BUDGET", m, span)
                    .label("a product of weights with a literal is a hand-built forward pass")
                    .hint(
                        "products of weights with weights are free; observation goes through run()",
                    )
            })?;
        }
        Ok(probe)
    }

    // ------------------------------------------------------------ calls

    pub fn call_value(
        &mut self,
        f: &FuncVal,
        args: Vec<Value>,
        nargout: usize,
        span: Span,
    ) -> R<Vec<Value>> {
        match f {
            FuncVal::Named(n) => self.call_named(n, args, nargout, span),
            FuncVal::TestMethod { context, method } => {
                crate::testing::call_method(self, context, *method, &args, span)
            }
            FuncVal::Lambda { params, body, env } => {
                if args.len() > params.len() {
                    return Err(err(
                        "E-ARGS",
                        format!(
                            "this function takes {} argument(s), got {}",
                            params.len(),
                            args.len()
                        ),
                        span,
                    ));
                }
                let mut e2 = (**env).clone();
                for (p, a) in params.iter().zip(args) {
                    e2.insert(p.clone(), a);
                }
                self.depth += 1;
                if self.depth > 256 {
                    self.depth -= 1;
                    return Err(err("E-RECURSION", "calls nested too deeply", span));
                }
                let r = self.eval_multi(&mut e2, body, nargout.max(1));
                self.depth -= 1;
                r
            }
        }
    }

    pub fn call_named(
        &mut self,
        name: &str,
        args: Vec<Value>,
        nargout: usize,
        span: Span,
    ) -> R<Vec<Value>> {
        if let Some(f) = self.funcs.get(name).cloned() {
            // native fast path for scalar functions
            if let Some(nat) = &self.native {
                if nat.has(name) && args.len() == f.params.len() {
                    let mut xs = Vec::with_capacity(args.len());
                    let mut tag = Tag::Lit;
                    let mut ok = true;
                    for a in &args {
                        match a {
                            Value::Arr(t) if t.is_scalar() => {
                                xs.push(t.data[0]);
                                tag = tag.max(t.tag);
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        let y = nat.call(name, &xs).unwrap();
                        self.native_calls += 1;
                        return Ok(vec![Value::Arr(
                            Tensor::scalar(y).with_tag(Tag::derived(tag)),
                        )]);
                    }
                }
            }
            return self.call_user(&f, args, nargout, span);
        }
        builtins::call(self, name, args, nargout, span)
    }

    fn call_user(
        &mut self,
        f: &Function,
        args: Vec<Value>,
        nargout: usize,
        span: Span,
    ) -> R<Vec<Value>> {
        if args.len() > f.params.len() {
            return Err(err(
                "E-ARGS",
                format!(
                    "`{}` takes {} argument(s), got {}",
                    f.name,
                    f.params.len(),
                    args.len()
                ),
                span,
            ));
        }
        self.depth += 1;
        if self.depth > 256 {
            self.depth -= 1;
            return Err(err("E-RECURSION", "calls nested too deeply", span));
        }
        let mut env = Env::new();
        for (p, a) in f.params.iter().zip(args) {
            env.insert(p.clone(), a);
        }
        let r = self.block(&mut env, &f.body);
        self.depth -= 1;
        r?;
        let want = nargout
            .max(if f.outs.is_empty() { 0 } else { 1 })
            .min(f.outs.len());
        let mut out = Vec::new();
        for o in &f.outs[..want] {
            match env.remove(o) {
                Some(v) => out.push(v),
                None => {
                    if nargout == 0 {
                        break;
                    }
                    return Err(err(
                        "E-UNDEFINED",
                        format!("`{}` never set its output `{o}`", f.name),
                        span,
                    ));
                }
            }
        }
        Ok(out)
    }
}

/// Write `v` into a part at `idxs` (part viewed with `shape`).
pub fn write_part(
    pd: &mut PartData,
    shape: &[usize],
    idxs: &[Idx],
    v: &Tensor,
) -> Result<(), String> {
    let pos = index::flat_positions(idxs, shape)?;
    let region = index::region_shape(idxs);
    let vals = index::values_for_region(v, &region)?;
    for (p, x) in pos.iter().zip(vals) {
        pd.data[*p] = x;
    }
    pd.max_tag = pd.max_tag.max(v.tag);
    Ok(())
}

pub fn net_combine(a: &NetRef, b: &NetRef, sub: bool) -> Result<NetData, String> {
    let a = a.borrow();
    let b = b.borrow();
    let mut out = a.clone();
    for (name, bp) in &b.parts {
        if !a.parts.contains_key(name) {
            return Err(format!(
                "the right net has a part `{name}` that the left net lacks"
            ));
        }
        let op = out.parts.get_mut(name).unwrap();
        if op.shape != bp.shape {
            return Err(format!(
                "part `{name}` has shape {} on the left and {} on the right",
                crate::tensor::shape_str(&op.shape),
                crate::tensor::shape_str(&bp.shape)
            ));
        }
        for (x, y) in op.data.iter_mut().zip(&bp.data) {
            if sub {
                *x -= y;
            } else {
                *x += y;
            }
        }
        op.max_tag = op.max_tag.max(bp.max_tag).max(Tag::Alg);
    }
    Ok(out)
}

pub fn net_scale(n: &NetRef, k: f64, tag: Tag) -> NetData {
    let mut out = n.borrow().clone();
    for p in out.parts.values_mut() {
        for x in p.data.iter_mut() {
            *x *= k;
        }
        p.max_tag = p.max_tag.max(tag).max(Tag::Alg);
    }
    out
}

pub fn closest<'a>(name: &str, cands: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in cands {
        let d = levenshtein(name, c);
        let limit = (name.len().max(2) / 3).max(1);
        if d <= limit && best.map(|b| d < b.0).unwrap_or(true) {
            best = Some((d, c));
        }
    }
    best.map(|b| b.1.to_string())
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c);
        }
        prev = cur;
    }
    prev[b.len()]
}

impl NetData {
    pub fn summary(&self) -> String {
        if let Some(model) = &self.decoder {
            return format!("net `{}`: {}", self.name, model.summary());
        }
        let arch = match &self.arch {
            crate::net::Arch::Mlp { sizes, .. } => format!(
                "mlp {}",
                sizes
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            ),
            crate::net::Arch::Transformer(c) => format!(
                "transformer d={} layers={} heads={} ff={} vocab={}",
                c.d,
                c.layers,
                c.heads,
                c.ff,
                c.vocab.len()
            ),
            crate::net::Arch::Unknown => "unknown architecture".into(),
        };
        format!(
            "net `{}`: {arch}, {} parts, {} weights",
            self.name,
            self.parts.len(),
            self.total_weights()
        )
    }
}

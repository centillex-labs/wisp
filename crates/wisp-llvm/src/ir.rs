//! LLVM IR generation.
//!
//! Two code generators:
//! * `scalar_module` compiles Wisp functions that only use scalar numbers
//!   (arithmetic, comparisons, if, for over ranges, while, math built-ins and
//!   calls to other such functions) into native functions.
//! * `mlp_forward` compiles a dense network's forward pass, with the weights
//!   embedded as constants, into `void wisp_forward(const double*, double*)`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;
use wisp_syntax::ast::*;

pub mod native;

fn konst(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

const NAN: &str = "0x7FF8000000000000";

const MATH1: &[&str] = &[
    "abs", "sqrt", "exp", "log", "floor", "ceil", "round", "tanh", "sin", "cos", "relu", "sigmoid",
    "sign",
];
const MATH2: &[&str] = &["mod", "rem", "min", "max", "atan2"];

pub struct ScalarModule {
    pub ir: String,
    /// name -> number of parameters
    pub compiled: Vec<(String, usize)>,
    /// name -> why it could not be compiled
    pub rejected: Vec<(String, String)>,
}

/// Compile every function in `funcs` that is purely scalar.
pub fn scalar_module(funcs: &[&Function]) -> ScalarModule {
    let by_name: HashMap<&str, &Function> = funcs.iter().map(|f| (f.name.as_str(), *f)).collect();
    let mut candidates: HashSet<String> = HashSet::new();
    let mut rejected: Vec<(String, String)> = Vec::new();
    for f in funcs {
        match check_fn(f) {
            Ok(()) => {
                candidates.insert(f.name.clone());
            }
            Err(e) => rejected.push((f.name.clone(), e)),
        }
    }
    // remove candidates that call non-candidates, until stable
    loop {
        let mut drop = Vec::new();
        for name in &candidates {
            let f = by_name[name.as_str()];
            for callee in calls_of(f) {
                if MATH1.contains(&callee.as_str()) || MATH2.contains(&callee.as_str()) {
                    continue;
                }
                if !candidates.contains(&callee) {
                    drop.push((
                        name.clone(),
                        format!("calls `{callee}`, which cannot be compiled"),
                    ));
                    break;
                }
            }
        }
        if drop.is_empty() {
            break;
        }
        for (n, why) in drop {
            candidates.remove(&n);
            rejected.push((n, why));
        }
    }
    let mut ir = String::new();
    ir += "; Wisp scalar functions\n";
    for d in [
        "declare double @llvm.pow.f64(double, double)",
        "declare double @llvm.fabs.f64(double)",
        "declare double @llvm.sqrt.f64(double)",
        "declare double @llvm.exp.f64(double)",
        "declare double @llvm.log.f64(double)",
        "declare double @llvm.floor.f64(double)",
        "declare double @llvm.ceil.f64(double)",
        "declare double @llvm.round.f64(double)",
        "declare double @llvm.sin.f64(double)",
        "declare double @llvm.cos.f64(double)",
        "declare double @llvm.minnum.f64(double, double)",
        "declare double @llvm.maxnum.f64(double, double)",
        "declare double @tanh(double)",
        "declare double @atan2(double, double)",
        "declare double @fmod(double, double)",
    ] {
        ir += d;
        ir += "\n";
    }
    let mut compiled = Vec::new();
    let mut names: Vec<&String> = candidates.iter().collect();
    names.sort();
    for name in names {
        let f = by_name[name.as_str()];
        let mut g = FnGen::new(f);
        match g.function() {
            Ok(text) => {
                ir += &text;
                compiled.push((f.name.clone(), f.params.len()));
            }
            Err(e) => rejected.push((f.name.clone(), e)),
        }
    }
    ScalarModule {
        ir,
        compiled,
        rejected,
    }
}

fn calls_of(f: &Function) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    fn ex(e: &Expr, out: &mut BTreeSet<String>) {
        match &e.kind {
            ExprKind::Postfix(b, Post::Call(args)) => {
                if let ExprKind::Ident(n) = &b.kind {
                    out.insert(n.clone());
                }
                for a in args {
                    ex(&a.value, out);
                }
            }
            ExprKind::Unary(_, a) => ex(a, out),
            ExprKind::Binary(_, a, b) => {
                ex(a, out);
                ex(b, out);
            }
            ExprKind::Range(a, b, c) => {
                ex(a, out);
                ex(b, out);
                if let Some(c) = c {
                    ex(c, out);
                }
            }
            _ => {}
        }
    }
    fn st(s: &[Stmt], out: &mut BTreeSet<String>) {
        for s in s {
            match &s.kind {
                StmtKind::Assign(_, _, e) | StmtKind::Expr(e) => ex(e, out),
                StmtKind::If(arms, el) => {
                    for (c, b) in arms {
                        ex(c, out);
                        st(b, out);
                    }
                    if let Some(b) = el {
                        st(b, out);
                    }
                }
                StmtKind::For(_, e, b) | StmtKind::While(e, b) => {
                    ex(e, out);
                    st(b, out);
                }
                _ => {}
            }
        }
    }
    st(&f.body, &mut out);
    out
}

/// Structural check: only scalar constructs.
fn check_fn(f: &Function) -> Result<(), String> {
    if f.outs.len() != 1 {
        return Err("needs exactly one output".into());
    }
    let mut vars: HashSet<String> = f.params.iter().cloned().collect();
    collect_assigned(&f.body, &mut vars);
    vars.insert(f.outs[0].clone());
    fn ex(e: &Expr, vars: &HashSet<String>) -> Result<(), String> {
        match &e.kind {
            ExprKind::Num(_) | ExprKind::Bool(_) => Ok(()),
            ExprKind::Ident(n) => {
                if vars.contains(n) || n == "pi" {
                    Ok(())
                } else {
                    Err(format!("uses `{n}`, which is not a local scalar"))
                }
            }
            ExprKind::Unary(_, a) => ex(a, vars),
            ExprKind::Binary(op, a, b) => {
                if *op == BinOp::LeftDiv {
                    return Err("uses `\\`".into());
                }
                ex(a, vars)?;
                ex(b, vars)
            }
            ExprKind::Postfix(b, Post::Call(args)) => {
                if !matches!(b.kind, ExprKind::Ident(_)) {
                    return Err("calls a function value".into());
                }
                for a in args {
                    if a.name.is_some() {
                        return Err("uses a named argument".into());
                    }
                    ex(&a.value, vars)?;
                }
                Ok(())
            }
            _ => Err("uses an array, string, cell, index or field".into()),
        }
    }
    fn st(s: &[Stmt], vars: &HashSet<String>) -> Result<(), String> {
        for s in s {
            match &s.kind {
                StmtKind::Assign(lv, _, e) => {
                    if !lv.path.is_empty() {
                        return Err("assigns into an index or field".into());
                    }
                    ex(e, vars)?;
                }
                StmtKind::If(arms, el) => {
                    for (c, b) in arms {
                        ex(c, vars)?;
                        st(b, vars)?;
                    }
                    if let Some(b) = el {
                        st(b, vars)?;
                    }
                }
                StmtKind::For(_, e, b) => {
                    match &e.kind {
                        ExprKind::Range(a, bb, c) => {
                            ex(a, vars)?;
                            ex(bb, vars)?;
                            if let Some(c) = c {
                                ex(c, vars)?;
                            }
                        }
                        _ => return Err("loops over something other than a range".into()),
                    }
                    st(b, vars)?;
                }
                StmtKind::While(c, b) => {
                    ex(c, vars)?;
                    st(b, vars)?;
                }
                StmtKind::Break | StmtKind::Continue | StmtKind::Return => {}
                StmtKind::Expr(_) => {
                    return Err("has an expression statement (output is not scalar-only)".into())
                }
                StmtKind::Multi(..) => return Err("uses a multi-assignment".into()),
            }
        }
        Ok(())
    }
    st(&f.body, &vars)
}

fn collect_assigned(s: &[Stmt], vars: &mut HashSet<String>) {
    for s in s {
        match &s.kind {
            StmtKind::Assign(lv, _, _) => {
                vars.insert(lv.base.clone());
            }
            StmtKind::If(arms, el) => {
                for (_, b) in arms {
                    collect_assigned(b, vars);
                }
                if let Some(b) = el {
                    collect_assigned(b, vars);
                }
            }
            StmtKind::For(v, _, b) => {
                vars.insert(v.clone());
                collect_assigned(b, vars);
            }
            StmtKind::While(_, b) => collect_assigned(b, vars),
            _ => {}
        }
    }
}

struct FnGen<'a> {
    f: &'a Function,
    body: String,
    tmp: usize,
    label: usize,
    vars: HashMap<String, String>,
    loops: Vec<(String, String)>,
    cur: String,
    terminated: bool,
    extra_allocas: String,
    strict: bool,
    globals: String,
}

impl<'a> FnGen<'a> {
    fn new(f: &'a Function) -> Self {
        FnGen {
            f,
            body: String::new(),
            tmp: 0,
            label: 0,
            vars: HashMap::new(),
            loops: vec![],
            cur: "entry".into(),
            terminated: false,
            extra_allocas: String::new(),
            strict: false,
            globals: String::new(),
        }
    }
    fn t(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp)
    }
    fn l(&mut self, base: &str) -> String {
        self.label += 1;
        format!("{base}{}", self.label)
    }
    fn emit(&mut self, s: &str) {
        self.body += "  ";
        self.body += s;
        self.body += "\n";
    }
    fn start(&mut self, label: &str) {
        self.body += &format!("{label}:\n");
        self.cur = label.to_string();
        self.terminated = false;
    }
    fn br(&mut self, label: &str) {
        if !self.terminated {
            self.emit(&format!("br label %{label}"));
            self.terminated = true;
        }
    }

    fn function(&mut self) -> Result<String, String> {
        let f = self.f;
        let mut names: HashSet<String> = f.params.iter().cloned().collect();
        collect_assigned(&f.body, &mut names);
        names.insert(f.outs[0].clone());
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        let mut head = String::new();
        let params: Vec<String> = (0..f.params.len())
            .map(|i| format!("double %p{i}"))
            .collect();
        writeln!(
            head,
            "\ndefine double @wisp_{}_impl({}) {{",
            f.name,
            params.join(", ")
        )
        .unwrap();
        head += "entry:\n";
        for (k, n) in names.iter().enumerate() {
            let slot = format!("%v{k}");
            writeln!(head, "  {slot} = alloca double").unwrap();
            if self.strict {
                writeln!(
                    head,
                    "  {slot}_set = alloca i1\n  store i1 {}, ptr {slot}_set",
                    f.params.contains(n)
                )
                .unwrap();
            }
            self.vars.insert(n.clone(), slot);
        }
        for n in &names {
            let slot = self.vars[n].clone();
            match f.params.iter().position(|p| p == n) {
                Some(i) => writeln!(head, "  store double %p{i}, ptr {slot}").unwrap(),
                None => writeln!(head, "  store double {NAN}, ptr {slot}").unwrap(),
            }
        }
        self.cur = "entry".into();
        let ret = "ret".to_string();
        let body = f.body.clone();
        self.stmts(&body, &ret)?;
        self.br(&ret);
        self.start("ret");
        let out_slot = self.vars[&f.outs[0]].clone();
        self.check_initialized(&out_slot, f.span);
        self.emit(&format!("%r = load double, ptr {out_slot}"));
        self.emit("ret double %r");
        let mut text = self.globals.clone()
            + &head.replacen("entry:\n", &format!("entry:\n{}", self.extra_allocas), 1)
            + &self.body;
        text += "}\n";
        // uniform entry point: double wisp_NAME(const double* args)
        writeln!(
            text,
            "\ndefine double @wisp_{}(ptr %args) {{\nentry:",
            f.name
        )
        .unwrap();
        let mut argv = Vec::new();
        for i in 0..f.params.len() {
            writeln!(text, "  %ap{i} = getelementptr inbounds double, ptr %args, i64 {i}\n  %a{i} = load double, ptr %ap{i}").unwrap();
            argv.push(format!("double %a{i}"));
        }
        writeln!(
            text,
            "  %r = call double @wisp_{}_impl({})\n  ret double %r\n}}",
            f.name,
            argv.join(", ")
        )
        .unwrap();
        Ok(text)
    }

    fn stmts(&mut self, s: &[Stmt], ret: &str) -> Result<(), String> {
        for st in s {
            if self.terminated {
                let dead = self.l("dead");
                self.start(&dead);
            }
            self.stmt(st, ret)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt, ret: &str) -> Result<(), String> {
        match &s.kind {
            StmtKind::Assign(lv, op, e) => {
                let v = self.expr(e)?;
                let slot = self.vars[&lv.base].clone();
                let v = match op {
                    AssignOp::Set => v,
                    _ => {
                        self.check_initialized(&slot, lv.span);
                        let cur = self.t();
                        self.emit(&format!("{cur} = load double, ptr {slot}"));
                        let ins = match op {
                            AssignOp::Add => "fadd",
                            AssignOp::Sub => "fsub",
                            AssignOp::Mul => "fmul",
                            _ => "fdiv",
                        };
                        let r = self.t();
                        self.emit(&format!("{r} = {ins} double {cur}, {v}"));
                        r
                    }
                };
                self.emit(&format!("store double {v}, ptr {slot}"));
                if self.strict {
                    self.emit(&format!("store i1 true, ptr {slot}_set"));
                }
            }
            StmtKind::Expr(e) if self.strict => self.native_statement(e)?,
            StmtKind::If(arms, el) => {
                let end = self.l("endif");
                for (c, body) in arms {
                    let v = self.expr(c)?;
                    let b = self.t();
                    self.emit(&format!("{b} = fcmp une double {v}, 0.0"));
                    let then = self.l("then");
                    let next = self.l("else");
                    self.emit(&format!("br i1 {b}, label %{then}, label %{next}"));
                    self.terminated = true;
                    self.start(&then);
                    self.stmts(body, ret)?;
                    self.br(&end);
                    self.start(&next);
                }
                if let Some(body) = el {
                    self.stmts(body, ret)?;
                }
                self.br(&end);
                self.start(&end);
            }
            StmtKind::For(var, e, body) => {
                let (a, b, step) = match &e.kind {
                    ExprKind::Range(a, b, s) => (a, b, s),
                    _ => return Err("for over a non-range".into()),
                };
                let av = self.expr(a)?;
                let bv = self.expr(b)?;
                let sv = match step {
                    Some(s) => self.expr(s)?,
                    None => "1.0".to_string(),
                };
                if self.strict {
                    let valid = self.t();
                    self.emit(&format!("{valid} = fcmp one double {sv}, 0.0"));
                    self.native_guard(&valid, 3, e.span);
                    for value in [&av, &bv, &sv] {
                        let abs = self.t();
                        self.emit(&format!(
                            "{abs} = call double @llvm.fabs.f64(double {value})"
                        ));
                        let finite = self.t();
                        self.emit(&format!(
                            "{finite} = fcmp ole double {abs}, {}",
                            konst(f64::MAX)
                        ));
                        self.native_guard(&finite, 3, e.span);
                    }
                }
                // n = floor((b - a) / s) + 1, at least 0
                let diff = self.t();
                self.emit(&format!("{diff} = fsub double {bv}, {av}"));
                let q = self.t();
                self.emit(&format!("{q} = fdiv double {diff}, {sv}"));
                let qe = self.t();
                self.emit(&format!("{qe} = fadd double {q}, 1.0e-9"));
                let fl = self.t();
                self.emit(&format!("{fl} = call double @llvm.floor.f64(double {qe})"));
                let n = self.t();
                self.emit(&format!("{n} = fadd double {fl}, 1.0"));
                if self.strict {
                    let valid = self.t();
                    self.emit(&format!(
                        "{valid} = fcmp ole double {n}, {}",
                        konst(9007199254740991.0)
                    ));
                    self.native_guard(&valid, 3, e.span);
                }
                let kslot = format!("%k{}", self.label + 1);
                // counter slot lives in the entry block, so nested loops do not grow the stack
                self.extra_allocas += &format!("  {kslot} = alloca double\n");
                self.emit(&format!("store double 0.0, ptr {kslot}"));
                let head = self.l("for_head");
                let bodyl = self.l("for_body");
                let latch = self.l("for_latch");
                let exit = self.l("for_exit");
                self.br(&head);
                self.start(&head);
                let k = self.t();
                self.emit(&format!("{k} = load double, ptr {kslot}"));
                let c = self.t();
                self.emit(&format!("{c} = fcmp olt double {k}, {n}"));
                self.emit(&format!("br i1 {c}, label %{bodyl}, label %{exit}"));
                self.terminated = true;
                self.start(&bodyl);
                let ks = self.t();
                self.emit(&format!("{ks} = fmul double {k}, {sv}"));
                let iv = self.t();
                self.emit(&format!("{iv} = fadd double {av}, {ks}"));
                let slot = self.vars[var].clone();
                self.emit(&format!("store double {iv}, ptr {slot}"));
                if self.strict {
                    self.emit(&format!("store i1 true, ptr {slot}_set"));
                }
                self.loops.push((latch.clone(), exit.clone()));
                self.stmts(body, ret)?;
                self.loops.pop();
                self.br(&latch);
                self.start(&latch);
                let k2 = self.t();
                self.emit(&format!("{k2} = load double, ptr {kslot}"));
                let k3 = self.t();
                self.emit(&format!("{k3} = fadd double {k2}, 1.0"));
                self.emit(&format!("store double {k3}, ptr {kslot}"));
                self.br(&head);
                self.start(&exit);
            }
            StmtKind::While(c, body) => {
                let head = self.l("while_head");
                let bodyl = self.l("while_body");
                let exit = self.l("while_exit");
                self.br(&head);
                self.start(&head);
                let v = self.expr(c)?;
                let b = self.t();
                self.emit(&format!("{b} = fcmp une double {v}, 0.0"));
                self.emit(&format!("br i1 {b}, label %{bodyl}, label %{exit}"));
                self.terminated = true;
                self.start(&bodyl);
                self.loops.push((head.clone(), exit.clone()));
                self.stmts(body, ret)?;
                self.loops.pop();
                self.br(&head);
                self.start(&exit);
            }
            StmtKind::Break => {
                let (_, exit) = self.loops.last().cloned().ok_or("break outside a loop")?;
                self.br(&exit);
            }
            StmtKind::Continue => {
                let (cont, _) = self
                    .loops
                    .last()
                    .cloned()
                    .ok_or("continue outside a loop")?;
                self.br(&cont);
            }
            StmtKind::Return => self.br(ret),
            _ => return Err("unsupported statement".into()),
        }
        Ok(())
    }

    fn bool_to_double(&mut self, b: &str) -> String {
        let r = self.t();
        self.emit(&format!("{r} = uitofp i1 {b} to double"));
        r
    }

    fn expr(&mut self, e: &Expr) -> Result<String, String> {
        Ok(match &e.kind {
            ExprKind::Num(v) => konst(*v),
            ExprKind::Bool(b) => konst(if *b { 1.0 } else { 0.0 }),
            ExprKind::Ident(n) if n == "pi" && !self.vars.contains_key(n) => {
                konst(std::f64::consts::PI)
            }
            ExprKind::Ident(n) => {
                let slot = self.vars.get(n).cloned().ok_or(format!("unknown `{n}`"))?;
                self.check_initialized(&slot, e.span);
                let r = self.t();
                self.emit(&format!("{r} = load double, ptr {slot}"));
                r
            }
            ExprKind::Unary(op, a) => {
                let v = self.expr(a)?;
                let r = self.t();
                match op {
                    UnOp::Neg => self.emit(&format!("{r} = fneg double {v}")),
                    UnOp::Plus => return Ok(v),
                    UnOp::Not => {
                        self.emit(&format!("{r} = fcmp oeq double {v}, 0.0"));
                        return Ok(self.bool_to_double(&r));
                    }
                }
                r
            }
            ExprKind::Binary(op @ (BinOp::And | BinOp::Or), a, b) => {
                let av = self.expr(a)?;
                let ab = self.t();
                self.emit(&format!("{ab} = fcmp une double {av}, 0.0"));
                let rhs = self.l("sc_rhs");
                let end = self.l("sc_end");
                let from = self.cur.clone();
                if *op == BinOp::And {
                    self.emit(&format!("br i1 {ab}, label %{rhs}, label %{end}"));
                } else {
                    self.emit(&format!("br i1 {ab}, label %{end}, label %{rhs}"));
                }
                self.terminated = true;
                self.start(&rhs);
                let bv = self.expr(b)?;
                let bb = self.t();
                self.emit(&format!("{bb} = fcmp une double {bv}, 0.0"));
                let bd = self.bool_to_double(&bb);
                let rhs_end = self.cur.clone();
                self.br(&end);
                self.start(&end);
                let r = self.t();
                let short = if *op == BinOp::And { "0.0" } else { "1.0" };
                self.emit(&format!(
                    "{r} = phi double [ {short}, %{from} ], [ {bd}, %{rhs_end} ]"
                ));
                r
            }
            ExprKind::Binary(op, a, b) => {
                let av = self.expr(a)?;
                let bv = self.expr(b)?;
                let r = self.t();
                let cmp = |c: &str| format!("{r} = fcmp {c} double {av}, {bv}");
                match op {
                    BinOp::Add => self.emit(&format!("{r} = fadd double {av}, {bv}")),
                    BinOp::Sub => self.emit(&format!("{r} = fsub double {av}, {bv}")),
                    BinOp::MatMul | BinOp::ElMul => {
                        self.emit(&format!("{r} = fmul double {av}, {bv}"))
                    }
                    BinOp::Div | BinOp::ElDiv => {
                        self.emit(&format!("{r} = fdiv double {av}, {bv}"))
                    }
                    BinOp::Pow | BinOp::ElPow => self.emit(&format!(
                        "{r} = call double @llvm.pow.f64(double {av}, double {bv})"
                    )),
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let c = match op {
                            BinOp::Eq => "oeq",
                            BinOp::Ne => "une",
                            BinOp::Lt => "olt",
                            BinOp::Le => "ole",
                            BinOp::Gt => "ogt",
                            _ => "oge",
                        };
                        self.emit(&cmp(c));
                        return Ok(self.bool_to_double(&r));
                    }
                    _ => return Err("unsupported operator".into()),
                }
                r
            }
            ExprKind::Postfix(b, Post::Call(args)) => {
                let name = match &b.kind {
                    ExprKind::Ident(n) => n.clone(),
                    _ => return Err("call of a non-name".into()),
                };
                let mut vs = Vec::new();
                for a in args {
                    vs.push(self.expr(&a.value)?);
                }
                let r = self.t();
                let one = |s: &mut Self, intr: &str| -> Result<(), String> {
                    if vs.len() != 1 {
                        return Err(format!("`{name}` takes one argument here"));
                    }
                    s.emit(&format!("{r} = call double @{intr}(double {})", vs[0]));
                    Ok(())
                };
                match name.as_str() {
                    "abs" => one(self, "llvm.fabs.f64")?,
                    "sqrt" => one(self, "llvm.sqrt.f64")?,
                    "exp" => one(self, "llvm.exp.f64")?,
                    "log" => one(self, "llvm.log.f64")?,
                    "floor" => one(self, "llvm.floor.f64")?,
                    "ceil" => one(self, "llvm.ceil.f64")?,
                    "round" => one(self, "llvm.round.f64")?,
                    "sin" => one(self, "llvm.sin.f64")?,
                    "cos" => one(self, "llvm.cos.f64")?,
                    "tanh" => one(self, "tanh")?,
                    "relu" => {
                        if vs.len() != 1 {
                            return Err("relu takes one argument".into());
                        }
                        self.emit(&format!(
                            "{r} = call double @llvm.maxnum.f64(double {}, double 0.0)",
                            vs[0]
                        ));
                    }
                    "sigmoid" => {
                        let n = self.t();
                        self.emit(&format!("{n} = fneg double {}", vs[0]));
                        let ex = self.t();
                        self.emit(&format!("{ex} = call double @llvm.exp.f64(double {n})"));
                        let d = self.t();
                        self.emit(&format!("{d} = fadd double {ex}, 1.0"));
                        self.emit(&format!("{r} = fdiv double 1.0, {d}"));
                    }
                    "sign" => {
                        let p = self.t();
                        self.emit(&format!("{p} = fcmp ogt double {}, 0.0", vs[0]));
                        let pd = self.bool_to_double(&p);
                        let n = self.t();
                        self.emit(&format!("{n} = fcmp olt double {}, 0.0", vs[0]));
                        let nd = self.bool_to_double(&n);
                        self.emit(&format!("{r} = fsub double {pd}, {nd}"));
                    }
                    "min" | "max" | "mod" | "rem" | "atan2" if vs.len() == 2 => {
                        let (x, y) = (&vs[0], &vs[1]);
                        match name.as_str() {
                            "min" => self.emit(&format!(
                                "{r} = call double @llvm.minnum.f64(double {x}, double {y})"
                            )),
                            "max" => self.emit(&format!(
                                "{r} = call double @llvm.maxnum.f64(double {x}, double {y})"
                            )),
                            "rem" => self
                                .emit(&format!("{r} = call double @fmod(double {x}, double {y})")),
                            "atan2" => self
                                .emit(&format!("{r} = call double @atan2(double {x}, double {y})")),
                            _ => {
                                // MATLAB mod: x - floor(x / y) * y
                                let q = self.t();
                                self.emit(&format!("{q} = fdiv double {x}, {y}"));
                                let fq = self.t();
                                self.emit(&format!(
                                    "{fq} = call double @llvm.floor.f64(double {q})"
                                ));
                                let m = self.t();
                                self.emit(&format!("{m} = fmul double {fq}, {y}"));
                                self.emit(&format!("{r} = fsub double {x}, {m}"));
                            }
                        }
                    }
                    _ => {
                        let argv: Vec<String> = vs.iter().map(|v| format!("double {v}")).collect();
                        self.emit(&format!(
                            "{r} = call double @wisp_{name}_impl({})",
                            argv.join(", ")
                        ));
                        if self.strict {
                            let code = self.t();
                            self.emit(&format!("{code} = load i64, ptr @wisp_native_error"));
                            let ok = self.t();
                            self.emit(&format!("{ok} = icmp eq i64 {code}, 0"));
                            self.native_guard(&ok, 0, e.span);
                        }
                    }
                }
                r
            }
            _ => return Err("unsupported expression".into()),
        })
    }
}

/// One dense layer: `y = act(W x + b)`, W stored `[out, in]` row-major.
pub struct Dense {
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub n_out: usize,
    pub n_in: usize,
    pub act: String,
}

fn const_array(name: &str, v: &[f64]) -> String {
    if v.iter().all(|&x| x == 0.0) {
        return format!(
            "@{name} = private unnamed_addr constant [{} x double] zeroinitializer\n",
            v.len()
        );
    }
    let items: Vec<String> = v.iter().map(|&x| format!("double {}", konst(x))).collect();
    format!(
        "@{name} = private unnamed_addr constant [{} x double] [{}]\n",
        v.len(),
        items.join(", ")
    )
}

/// `void wisp_forward(const double* x, double* y)` for a stack of dense layers.
pub fn mlp_forward(layers: &[Dense]) -> Result<String, String> {
    if layers.is_empty() {
        return Err("no layers".into());
    }
    let mut ir = String::from("; Wisp MLP forward pass\ndeclare double @tanh(double)\ndeclare double @llvm.maxnum.f64(double, double)\ndeclare double @llvm.exp.f64(double)\n");
    for (l, d) in layers.iter().enumerate() {
        if d.w.len() != d.n_out * d.n_in || d.b.len() != d.n_out {
            return Err(format!("layer {l}: weight or bias size does not match"));
        }
        ir += &const_array(&format!("w{l}"), &d.w);
        ir += &const_array(&format!("b{l}"), &d.b);
    }
    let maxw = layers.iter().map(|d| d.n_out).max().unwrap();
    let n_in = layers[0].n_in;
    let n_out = layers.last().unwrap().n_out;
    writeln!(ir, "\ndefine i64 @wisp_n_in() {{\n  ret i64 {n_in}\n}}").unwrap();
    writeln!(ir, "define i64 @wisp_n_out() {{\n  ret i64 {n_out}\n}}").unwrap();
    ir += "\ndefine void @wisp_forward(ptr noalias %x, ptr noalias %y) {\nentry:\n";
    writeln!(ir, "  %bufA = alloca [{maxw} x double]\n  %bufB = alloca [{maxw} x double]\n  br label %L0_start").unwrap();
    let nl = layers.len();
    for (l, d) in layers.iter().enumerate() {
        let src = if l == 0 {
            "%x".to_string()
        } else if l % 2 == 1 {
            "%bufA".into()
        } else {
            "%bufB".into()
        };
        let dst = if l == nl - 1 {
            "%y".to_string()
        } else if l % 2 == 0 {
            "%bufA".into()
        } else {
            "%bufB".into()
        };
        let (no, ni) = (d.n_out, d.n_in);
        let s = format!(".l{l}");
        let act = match d.act.as_str() {
            "tanh" => format!("  %a{s} = call double @tanh(double %acc{s})\n"),
            "relu" => format!("  %a{s} = call double @llvm.maxnum.f64(double %acc{s}, double 0.0)\n"),
            "sigmoid" => format!(
                "  %ng{s} = fneg double %acc{s}\n  %ex{s} = call double @llvm.exp.f64(double %ng{s})\n  %de{s} = fadd double %ex{s}, 1.0\n  %a{s} = fdiv double 1.0, %de{s}\n"
            ),
            "linear" => format!("  %a{s} = fadd double %acc{s}, 0.0\n"),
            other => return Err(format!("activation `{other}` is not supported by the LLVM backend")),
        };
        write!(
            ir,
            "L{l}_start:\n  br label %L{l}_oh\n\
L{l}_oh:\n  %o{s} = phi i64 [ 0, %L{l}_start ], [ %on{s}, %L{l}_olatch ]\n  %oc{s} = icmp ult i64 %o{s}, {no}\n  br i1 %oc{s}, label %L{l}_ob, label %L{l}_done\n\
L{l}_ob:\n  %bp{s} = getelementptr inbounds [{no} x double], ptr @b{l}, i64 0, i64 %o{s}\n  %bv{s} = load double, ptr %bp{s}\n  %row{s} = mul i64 %o{s}, {ni}\n  br label %L{l}_ih\n\
L{l}_ih:\n  %i{s} = phi i64 [ 0, %L{l}_ob ], [ %in{s}, %L{l}_ib ]\n  %acc{s} = phi double [ %bv{s}, %L{l}_ob ], [ %accn{s}, %L{l}_ib ]\n  %ic{s} = icmp ult i64 %i{s}, {ni}\n  br i1 %ic{s}, label %L{l}_ib, label %L{l}_iend\n\
L{l}_ib:\n  %wi{s} = add i64 %row{s}, %i{s}\n  %wp{s} = getelementptr inbounds [{wn} x double], ptr @w{l}, i64 0, i64 %wi{s}\n  %wv{s} = load double, ptr %wp{s}\n  %xp{s} = getelementptr inbounds double, ptr {src}, i64 %i{s}\n  %xv{s} = load double, ptr %xp{s}\n  %pr{s} = fmul double %wv{s}, %xv{s}\n  %accn{s} = fadd double %acc{s}, %pr{s}\n  %in{s} = add i64 %i{s}, 1\n  br label %L{l}_ih\n\
L{l}_iend:\n{act}  %dp{s} = getelementptr inbounds double, ptr {dst}, i64 %o{s}\n  store double %a{s}, ptr %dp{s}\n  br label %L{l}_olatch\n\
L{l}_olatch:\n  %on{s} = add i64 %o{s}, 1\n  br label %L{l}_oh\n\
L{l}_done:\n",
            wn = no * ni
        )
        .unwrap();
        if l + 1 < nl {
            writeln!(ir, "  br label %L{}_start", l + 1).unwrap();
        } else {
            ir += "  ret void\n";
        }
    }
    ir += "}\n";
    Ok(ir)
}

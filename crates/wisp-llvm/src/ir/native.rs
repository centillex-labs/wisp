//! Strict whole-script scalar compilation. No interpreter fallback or runtime
//! value dispatch: every accepted statement becomes LLVM IR before execution.
use super::*;
use wisp_syntax::{Diag, Span};

fn unsupported(span: Span, message: impl Into<String>) -> Diag {
    Diag::new("E-JIT-UNSUPPORTED", message, span)
        .hint("native JIT supports scalar programs only; use `wisp run file.wisp --interpret` explicitly for tensors and networks")
}

struct Check<'a> {
    functions: &'a HashMap<String, &'a Function>,
    locals: HashSet<String>,
    calls: BTreeSet<String>,
}
impl Check<'_> {
    fn expr(&mut self, e: &Expr) -> Result<(), Diag> {
        match &e.kind {
            ExprKind::Num(_) | ExprKind::Bool(_) => Ok(()),
            ExprKind::Ident(n) if self.locals.contains(n) || n == "pi" => Ok(()),
            ExprKind::Unary(_, a) => self.expr(a),
            ExprKind::Binary(op, a, b) if *op != BinOp::LeftDiv => {
                self.expr(a)?;
                self.expr(b)
            }
            ExprKind::Postfix(base, Post::Call(args)) => {
                let ExprKind::Ident(name) = &base.kind else {
                    return Err(unsupported(
                        e.span,
                        "native JIT cannot call function values",
                    ));
                };
                if self.locals.contains(name) {
                    return Err(unsupported(
                        e.span,
                        format!("call target `{name}` is shadowed by a local variable"),
                    ));
                }
                let arity = if let Some(f) = self.functions.get(name) {
                    self.calls.insert(name.clone());
                    f.params.len()
                } else if MATH1.contains(&name.as_str()) {
                    1
                } else if MATH2.contains(&name.as_str()) {
                    2
                } else {
                    return Err(unsupported(
                        e.span,
                        format!("call to `{name}` is not supported by native JIT"),
                    ));
                };
                if args.len() != arity {
                    return Err(Diag::new(
                        "E-ARGS",
                        format!("`{name}` needs {arity} argument(s), got {}", args.len()),
                        e.span,
                    ));
                }
                for arg in args {
                    if arg.name.is_some() {
                        return Err(unsupported(e.span, "named arguments are not supported"));
                    }
                    self.expr(&arg.value)?;
                }
                Ok(())
            }
            _ => Err(unsupported(
                e.span,
                "expression is not supported by scalar native JIT",
            )),
        }
    }
    fn block(&mut self, body: &[Stmt], loops: usize) -> Result<(), Diag> {
        for s in body {
            match &s.kind {
                StmtKind::Assign(lv, _, rhs) if lv.path.is_empty() => self.expr(rhs)?,
                StmtKind::Expr(e) => {
                    if let ExprKind::Postfix(base, Post::Call(args)) = &e.kind {
                        if let ExprKind::Ident(name) = &base.kind {
                            if name == "disp" || name == "assert" {
                                if args.len() != 1 || args[0].name.is_some() {
                                    return Err(unsupported(
                                        e.span,
                                        format!("native `{name}` takes exactly one argument"),
                                    ));
                                }
                                if name == "disp" {
                                    if let ExprKind::Str(text) = &args[0].value.kind {
                                        if text.contains('\0') {
                                            return Err(unsupported(
                                                e.span,
                                                "native literal output cannot contain NUL bytes",
                                            ));
                                        }
                                        continue;
                                    }
                                }
                                self.expr(&args[0].value)?;
                                continue;
                            }
                        }
                    }
                    self.expr(e)?;
                }
                StmtKind::If(arms, otherwise) => {
                    for (test, body) in arms {
                        self.expr(test)?;
                        self.block(body, loops)?;
                    }
                    if let Some(body) = otherwise {
                        self.block(body, loops)?;
                    }
                }
                StmtKind::For(_, range, body) => {
                    let ExprKind::Range(a, b, step) = &range.kind else {
                        return Err(unsupported(
                            range.span,
                            "native for loops require an inclusive range",
                        ));
                    };
                    self.expr(a)?;
                    self.expr(b)?;
                    if let Some(step) = step {
                        self.expr(step)?;
                    }
                    self.block(body, loops + 1)?;
                }
                StmtKind::While(test, body) => {
                    self.expr(test)?;
                    self.block(body, loops + 1)?;
                }
                StmtKind::Break | StmtKind::Continue if loops > 0 => {}
                StmtKind::Return => {}
                _ => {
                    return Err(unsupported(
                        s.span,
                        "statement is not supported by scalar native JIT",
                    ))
                }
            }
        }
        Ok(())
    }
}

/// Generate one native module for the entire script and all functions. Rejects
/// unsupported syntax, invalid call arity, and recursion before any side effects.
pub fn program(prog: &Program) -> Result<String, Diag> {
    let mut functions = HashMap::new();
    for f in &prog.functions {
        if f.name.starts_with("__wisp_")
            || MATH1.contains(&f.name.as_str())
            || MATH2.contains(&f.name.as_str())
            || ["disp", "assert"].contains(&f.name.as_str())
        {
            return Err(unsupported(
                f.span,
                format!("reserved native function name `{}`", f.name),
            ));
        }
        if functions.insert(f.name.clone(), f.as_ref()).is_some() {
            return Err(unsupported(f.span, "duplicate function definition"));
        }
        if f.outs.len() != 1 {
            return Err(unsupported(
                f.span,
                "native functions must declare exactly one scalar output",
            ));
        }
        if f.params.iter().collect::<HashSet<_>>().len() != f.params.len() {
            return Err(unsupported(f.span, "duplicate function parameter"));
        }
    }
    let span = prog
        .body
        .first()
        .map(|s| s.span)
        .unwrap_or(Span { start: 0, end: 0 });
    let output = "__wisp_result".to_string();
    let mut body = vec![Stmt {
        span,
        kind: StmtKind::Assign(
            LValue {
                base: output.clone(),
                path: vec![],
                span,
            },
            AssignOp::Set,
            Expr {
                kind: ExprKind::Num(0.0),
                span,
            },
        ),
    }];
    body.extend(prog.body.clone());
    let main = Function {
        name: "__wisp_script".into(),
        params: vec![],
        outs: vec![output],
        body,
        span,
    };
    let mut edges = HashMap::new();
    for f in prog
        .functions
        .iter()
        .map(|f| f.as_ref())
        .chain(std::iter::once(&main))
    {
        let mut locals: HashSet<_> = f.params.iter().chain(&f.outs).cloned().collect();
        collect_assigned(&f.body, &mut locals);
        if locals.contains("disp") || locals.contains("assert") {
            return Err(unsupported(
                f.span,
                "native output/assertion names cannot be shadowed",
            ));
        }
        let mut check = Check {
            functions: &functions,
            locals,
            calls: BTreeSet::new(),
        };
        check.block(&f.body, 0)?;
        edges.insert(f.name.clone(), check.calls);
    }
    fn visit(
        name: &str,
        edges: &HashMap<String, BTreeSet<String>>,
        active: &mut HashSet<String>,
        done: &mut HashSet<String>,
    ) -> bool {
        if done.contains(name) {
            return true;
        }
        if !active.insert(name.into()) {
            return false;
        }
        for callee in &edges[name] {
            if !visit(callee, edges, active, done) {
                return false;
            }
        }
        active.remove(name);
        done.insert(name.into());
        true
    }
    let mut done = HashSet::new();
    for f in &prog.functions {
        if !visit(&f.name, &edges, &mut HashSet::new(), &mut done) {
            return Err(unsupported(
                f.span,
                "recursive calls are not supported by native JIT yet",
            ));
        }
    }
    // Reuse the intrinsic declarations, but compile all bodies in strict mode.
    let mut ir = scalar_module(&[]).ir;
    ir += "\n@wisp_native_error = internal global i64 0\n@wisp_native_start = internal global i64 0\n@wisp_native_end = internal global i64 0\ndeclare i32 @printf(ptr, ...)\ndeclare i32 @fflush(ptr)\n";
    for f in prog
        .functions
        .iter()
        .map(|f| f.as_ref())
        .chain(std::iter::once(&main))
    {
        let mut gen = FnGen::new(f);
        gen.strict = true;
        ir += &gen.function().map_err(|m| unsupported(f.span, m))?;
    }
    ir += "\ndefine void @wisp_native_entry(ptr %result) {\nentry:\n  store i64 0, ptr @wisp_native_error\n  %unused = call double @wisp___wisp_script_impl()\n  %flush = call i32 @fflush(ptr null)\n  %code = load i64, ptr @wisp_native_error\n  %start = load i64, ptr @wisp_native_start\n  %end = load i64, ptr @wisp_native_end\n  store i64 %code, ptr %result\n  %p1 = getelementptr i64, ptr %result, i64 1\n  %p2 = getelementptr i64, ptr %result, i64 2\n  store i64 %start, ptr %p1\n  store i64 %end, ptr %p2\n  ret void\n}\n";
    Ok(ir)
}

impl FnGen<'_> {
    /// A failing guard returns directly from native code. Code 0 propagates an
    /// existing callee failure, preserving the original source span.
    pub(super) fn native_guard(&mut self, valid: &str, code: u64, span: Span) {
        let ok = self.l("native_ok");
        let bad = self.l("native_fail");
        self.emit(&format!("br i1 {valid}, label %{ok}, label %{bad}"));
        self.start(&bad);
        if code != 0 {
            self.emit(&format!("store i64 {code}, ptr @wisp_native_error"));
            self.emit(&format!("store i64 {}, ptr @wisp_native_start", span.start));
            self.emit(&format!("store i64 {}, ptr @wisp_native_end", span.end));
        }
        self.emit(&format!("ret double {NAN}"));
        self.start(&ok);
    }
    pub(super) fn check_initialized(&mut self, slot: &str, span: Span) {
        if !self.strict {
            return;
        }
        let valid = self.t();
        self.emit(&format!("{valid} = load i1, ptr {slot}_set"));
        self.native_guard(&valid, 2, span);
    }
    fn native_string(&mut self, s: &str) -> String {
        let name = format!("@wisp_text_{}_{}", self.f.name, self.tmp);
        self.tmp += 1;
        let bytes: String = s
            .as_bytes()
            .iter()
            .chain(std::iter::once(&0u8))
            .map(|b| format!("\\{b:02X}"))
            .collect();
        self.globals += &format!(
            "{name} = private constant [{} x i8] c\"{bytes}\"\n",
            s.len() + 1
        );
        name
    }
    pub(super) fn native_statement(&mut self, e: &Expr) -> Result<(), String> {
        if let ExprKind::Postfix(base, Post::Call(args)) = &e.kind {
            if let ExprKind::Ident(name) = &base.kind {
                if name == "disp" {
                    if let ExprKind::Str(s) = &args[0].value.kind {
                        let fmt = self.native_string("%s\n");
                        let text = self.native_string(s);
                        self.emit(&format!(
                            "call i32 (ptr, ...) @printf(ptr {fmt}, ptr {text})"
                        ));
                    } else {
                        let value = self.expr(&args[0].value)?;
                        let fmt = self.native_string("%.15g\n");
                        self.emit(&format!(
                            "call i32 (ptr, ...) @printf(ptr {fmt}, double {value})"
                        ));
                    }
                    return Ok(());
                }
                if name == "assert" {
                    let value = self.expr(&args[0].value)?;
                    let ok = self.t();
                    self.emit(&format!("{ok} = fcmp une double {value}, 0.0"));
                    self.native_guard(&ok, 1, e.span);
                    return Ok(());
                }
            }
        }
        self.expr(e)?;
        Ok(())
    }
}

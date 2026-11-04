//! Recursive-descent parser with precedence climbing for expressions.
//!
//! Newlines end statements, are ignored inside `( )` and index brackets, and
//! separate rows inside matrix and cell literals. Inside a literal, the two
//! whitespace rules from the specification decide where elements split.

use crate::ast::*;
use crate::diag::{Diag, Span};
use crate::lexer::{Tok, Token};
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    Stmt,
    Paren,
    Matrix,
}

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
    ctx: Vec<Ctx>,
}

type PResult<T> = Result<T, Diag>;

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser {
            toks,
            pos: 0,
            ctx: vec![Ctx::Stmt],
        }
    }

    // ---------------------------------------------------------------- helpers

    fn top(&self) -> Ctx {
        *self.ctx.last().unwrap()
    }

    fn skip_nl_if_paren(&mut self) {
        if self.top() == Ctx::Paren {
            while self.toks[self.pos].tok == Tok::Newline {
                self.pos += 1;
            }
        }
    }

    fn peek(&mut self) -> &Token {
        self.skip_nl_if_paren();
        &self.toks[self.pos]
    }

    fn peek_tok(&mut self) -> Tok {
        self.peek().tok.clone()
    }

    /// The token after the current one (newlines skipped in paren context).
    fn peek2(&mut self) -> Token {
        self.skip_nl_if_paren();
        let mut j = self.pos + 1;
        if self.top() == Ctx::Paren {
            while self.toks[j].tok == Tok::Newline {
                j += 1;
            }
        }
        self.toks[j.min(self.toks.len() - 1)].clone()
    }

    fn bump(&mut self) -> Token {
        self.skip_nl_if_paren();
        let t = self.toks[self.pos].clone();
        if t.tok != Tok::Eof {
            self.pos += 1;
        }
        t
    }

    fn at(&mut self, t: &Tok) -> bool {
        &self.peek().tok == t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: Tok, what: &str) -> PResult<Token> {
        if self.at(&t) {
            Ok(self.bump())
        } else {
            let tok = self.peek().clone();
            Err(Diag::new(
                "E-PARSE",
                format!("expected {what}, found {}", describe(&tok.tok)),
                tok.span,
            ))
        }
    }

    fn ident(&mut self, what: &str) -> PResult<(String, Span)> {
        let t = self.peek().clone();
        match t.tok {
            Tok::Ident(s) => {
                self.bump();
                Ok((s, t.span))
            }
            other => Err(Diag::new(
                "E-PARSE",
                format!("expected {what}, found {}", describe(&other)),
                t.span,
            )),
        }
    }

    fn with_ctx<T>(&mut self, c: Ctx, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        self.ctx.push(c);
        let r = f(self);
        self.ctx.pop();
        r
    }

    fn is_sep(t: &Tok) -> bool {
        matches!(t, Tok::Newline | Tok::Semi | Tok::Comma)
    }

    fn skip_seps(&mut self) {
        while Self::is_sep(&self.toks[self.pos].tok) {
            self.pos += 1;
        }
    }

    // ---------------------------------------------------------------- program

    pub fn program(&mut self) -> PResult<Program> {
        let mut prog = Program::default();
        loop {
            self.skip_seps();
            if self.at(&Tok::Eof) {
                break;
            }
            if self.at(&Tok::Function) {
                prog.functions.push(Rc::new(self.function()?));
            } else {
                let s = self.stmt()?;
                prog.body.push(s);
            }
        }
        Ok(prog)
    }

    fn function(&mut self) -> PResult<Function> {
        let start = self.expect(Tok::Function, "`function`")?.span;
        let mut outs = Vec::new();
        if self.at(&Tok::LBracket) {
            self.bump();
            loop {
                if self.eat(&Tok::RBracket) {
                    break;
                }
                let (n, _) = self.ident("an output name")?;
                outs.push(n);
                self.eat(&Tok::Comma);
            }
            self.expect(Tok::Assign, "`=` after the output list")?;
        } else if matches!(self.peek_tok(), Tok::Ident(_)) && self.peek2().tok == Tok::Assign {
            let (n, _) = self.ident("an output name")?;
            outs.push(n);
            self.bump(); // =
        }
        let (name, _) = self.ident("a function name")?;
        let mut params = Vec::new();
        self.expect(Tok::LParen, "`(`")?;
        self.with_ctx(Ctx::Paren, |p| {
            if !p.at(&Tok::RParen) {
                loop {
                    let (n, _) = p.ident("a parameter name")?;
                    params.push(n);
                    if !p.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            p.expect(Tok::RParen, "`)`")?;
            Ok(())
        })?;
        let body = self.block(&[Tok::End])?;
        let end = self.expect(Tok::End, "`end` to close the function")?.span;
        Ok(Function {
            name,
            params,
            outs,
            body,
            span: start.to(end),
        })
    }

    fn block(&mut self, terms: &[Tok]) -> PResult<Vec<Stmt>> {
        let mut out = Vec::new();
        loop {
            self.skip_seps();
            let t = self.peek_tok();
            if t == Tok::Eof || terms.contains(&t) {
                break;
            }
            if t == Tok::Function {
                let sp = self.peek().span;
                return Err(Diag::new(
                    "E-PARSE",
                    "a function cannot be defined inside a block",
                    sp,
                ));
            }
            out.push(self.stmt()?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- statements

    fn stmt(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let kind = match self.peek_tok() {
            Tok::If => self.if_stmt()?,
            Tok::For => {
                self.bump();
                let (var, _) = self.ident("a loop variable")?;
                self.expect(Tok::Assign, "`=` in a for loop")?;
                let iter = self.expr()?;
                let body = self.block(&[Tok::End])?;
                self.expect(Tok::End, "`end` to close the for loop")?;
                StmtKind::For(var, iter, body)
            }
            Tok::While => {
                self.bump();
                let cond = self.expr()?;
                let body = self.block(&[Tok::End])?;
                self.expect(Tok::End, "`end` to close the while loop")?;
                StmtKind::While(cond, body)
            }
            Tok::Break => {
                self.bump();
                StmtKind::Break
            }
            Tok::Continue => {
                self.bump();
                StmtKind::Continue
            }
            Tok::Return => {
                self.bump();
                StmtKind::Return
            }
            Tok::LBracket if self.is_multi_assign() => self.multi_assign()?,
            _ => {
                let e = self.expr()?;
                let op = match self.peek_tok() {
                    Tok::Assign => Some(AssignOp::Set),
                    Tok::PlusEq => Some(AssignOp::Add),
                    Tok::MinusEq => Some(AssignOp::Sub),
                    Tok::StarEq => Some(AssignOp::Mul),
                    Tok::SlashEq => Some(AssignOp::Div),
                    _ => None,
                };
                match op {
                    Some(op) => {
                        self.bump();
                        let lv = to_lvalue(e)?;
                        let rhs = self.expr()?;
                        StmtKind::Assign(lv, op, rhs)
                    }
                    None => StmtKind::Expr(e),
                }
            }
        };
        let t = self.toks[self.pos].clone();
        if !(Self::is_sep(&t.tok) || matches!(t.tok, Tok::Eof | Tok::End | Tok::Else | Tok::Elseif))
        {
            return Err(Diag::new(
                "E-PARSE",
                format!(
                    "expected the end of the statement, found {}",
                    describe(&t.tok)
                ),
                t.span,
            )
            .hint("put a newline or `;` between statements"));
        }
        let end = self.toks[self.pos.saturating_sub(1)].span;
        Ok(Stmt {
            kind,
            span: start.to(end),
        })
    }

    fn if_stmt(&mut self) -> PResult<StmtKind> {
        self.expect(Tok::If, "`if`")?;
        let mut arms = Vec::new();
        let cond = self.expr()?;
        let body = self.block(&[Tok::End, Tok::Elseif, Tok::Else])?;
        arms.push((cond, body));
        let mut else_ = None;
        loop {
            if self.eat(&Tok::Elseif) {
                let c = self.expr()?;
                let b = self.block(&[Tok::End, Tok::Elseif, Tok::Else])?;
                arms.push((c, b));
            } else if self.eat(&Tok::Else) {
                else_ = Some(self.block(&[Tok::End])?);
            } else {
                break;
            }
        }
        self.expect(Tok::End, "`end` to close the if")?;
        Ok(StmtKind::If(arms, else_))
    }

    /// `[` ... matching `]` followed by `=` (not `==`).
    fn is_multi_assign(&self) -> bool {
        let mut depth = 0i32;
        let mut j = self.pos;
        while j < self.toks.len() {
            match self.toks[j].tok {
                Tok::LBracket | Tok::LParen | Tok::LBrace => depth += 1,
                Tok::RBracket | Tok::RParen | Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return self
                            .toks
                            .get(j + 1)
                            .map(|t| t.tok == Tok::Assign)
                            .unwrap_or(false);
                    }
                }
                Tok::Eof => return false,
                Tok::Newline if depth <= 1 => return false,
                _ => {}
            }
            j += 1;
        }
        false
    }

    fn multi_assign(&mut self) -> PResult<StmtKind> {
        self.expect(Tok::LBracket, "`[`")?;
        let mut targets = Vec::new();
        self.with_ctx(Ctx::Matrix, |p| {
            loop {
                if p.eat(&Tok::RBracket) {
                    break;
                }
                if p.eat(&Tok::Ident("_".into())) || p.eat(&Tok::Tilde) {
                    targets.push(None);
                } else {
                    let e = p.postfix()?;
                    targets.push(Some(to_lvalue(e)?));
                }
                p.eat(&Tok::Comma);
            }
            Ok(())
        })?;
        self.expect(Tok::Assign, "`=`")?;
        let rhs = self.expr()?;
        Ok(StmtKind::Multi(targets, rhs))
    }

    // ---------------------------------------------------------------- expressions

    pub fn expr(&mut self) -> PResult<Expr> {
        self.or()
    }

    fn bin(l: Expr, op: BinOp, r: Expr) -> Expr {
        let span = l.span.to(r.span);
        Expr {
            kind: ExprKind::Binary(op, Box::new(l), Box::new(r)),
            span,
        }
    }

    fn or(&mut self) -> PResult<Expr> {
        let mut l = self.and()?;
        while self.eat(&Tok::OrOr) {
            let r = self.and()?;
            l = Self::bin(l, BinOp::Or, r);
        }
        Ok(l)
    }

    fn and(&mut self) -> PResult<Expr> {
        let mut l = self.cmp()?;
        while self.eat(&Tok::AndAnd) {
            let r = self.cmp()?;
            l = Self::bin(l, BinOp::And, r);
        }
        Ok(l)
    }

    fn cmp(&mut self) -> PResult<Expr> {
        let mut l = self.range()?;
        loop {
            let op = match self.peek_tok() {
                Tok::EqEq => BinOp::Eq,
                Tok::NotEq => BinOp::Ne,
                Tok::Lt => BinOp::Lt,
                Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.bump();
            let r = self.range()?;
            l = Self::bin(l, op, r);
        }
        Ok(l)
    }

    fn range(&mut self) -> PResult<Expr> {
        let a = self.add()?;
        if self.eat(&Tok::DotDot) {
            let b = self.add()?;
            let step = if self.eat(&Tok::By) {
                Some(Box::new(self.add()?))
            } else {
                None
            };
            let end = step.as_ref().map(|s| s.span).unwrap_or(b.span);
            let span = a.span.to(end);
            return Ok(Expr {
                kind: ExprKind::Range(Box::new(a), Box::new(b), step),
                span,
            });
        }
        Ok(a)
    }

    /// In a matrix literal, `a -b` starts a new element while `a - b` and `a-b` subtract.
    fn splits_element(&mut self) -> bool {
        if self.top() != Ctx::Matrix {
            return false;
        }
        let op = self.peek().clone();
        let next = self.peek2();
        op.space_before && !next.space_before
    }

    fn add(&mut self) -> PResult<Expr> {
        let mut l = self.mul()?;
        loop {
            let op = match self.peek_tok() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            if self.splits_element() {
                break;
            }
            self.bump();
            let r = self.mul()?;
            l = Self::bin(l, op, r);
        }
        Ok(l)
    }

    fn mul(&mut self) -> PResult<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek_tok() {
                Tok::Star => BinOp::MatMul,
                Tok::Slash => BinOp::Div,
                Tok::Backslash => BinOp::LeftDiv,
                Tok::DotStar => BinOp::ElMul,
                Tok::DotSlash => BinOp::ElDiv,
                _ => break,
            };
            self.bump();
            let r = self.unary()?;
            l = Self::bin(l, op, r);
        }
        Ok(l)
    }

    fn unary(&mut self) -> PResult<Expr> {
        let t = self.peek().clone();
        let op = match t.tok {
            Tok::Minus => Some(UnOp::Neg),
            Tok::Plus => Some(UnOp::Plus),
            Tok::Bang | Tok::Tilde => Some(UnOp::Not),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let e = self.unary()?;
            let span = t.span.to(e.span);
            return Ok(Expr {
                kind: ExprKind::Unary(op, Box::new(e)),
                span,
            });
        }
        self.power()
    }

    fn power(&mut self) -> PResult<Expr> {
        let base = self.postfix()?;
        let op = match self.peek_tok() {
            Tok::Caret => BinOp::Pow,
            Tok::DotCaret => BinOp::ElPow,
            _ => return Ok(base),
        };
        self.bump();
        let e = self.unary()?;
        Ok(Self::bin(base, op, e))
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            let t = self.peek().clone();
            let in_matrix = self.top() == Ctx::Matrix;
            let post = match t.tok {
                Tok::LParen if !(in_matrix && t.space_before) => {
                    self.bump();
                    Post::Call(self.args(Tok::RParen, false)?)
                }
                Tok::LBracket if !(in_matrix && t.space_before) => {
                    self.bump();
                    Post::Index(self.args(Tok::RBracket, true)?)
                }
                Tok::LBrace if !(in_matrix && t.space_before) => {
                    self.bump();
                    Post::CellIndex(self.args(Tok::RBrace, false)?)
                }
                Tok::Dot => {
                    self.bump();
                    if self.at(&Tok::LParen) {
                        self.bump();
                        let inner = self.with_ctx(Ctx::Paren, |p| {
                            let x = p.expr()?;
                            p.expect(Tok::RParen, "`)`")?;
                            Ok(x)
                        })?;
                        Post::DynField(Box::new(inner))
                    } else {
                        let (n, _) = self.ident("a field name after `.`")?;
                        Post::Field(n)
                    }
                }
                Tok::Transpose | Tok::DotTranspose => {
                    self.bump();
                    Post::Transpose
                }
                _ => break,
            };
            let end = self.toks[self.pos.saturating_sub(1)].span;
            let span = e.span.to(end);
            e = Expr {
                kind: ExprKind::Postfix(Box::new(e), post),
                span,
            };
        }
        Ok(e)
    }

    fn args(&mut self, close: Tok, allow_named: bool) -> PResult<Vec<Arg>> {
        self.with_ctx(Ctx::Paren, |p| {
            let mut out = Vec::new();
            if p.eat(&close) {
                return Ok(out);
            }
            loop {
                let t = p.peek().clone();
                let next = p.peek2();
                if t.tok == Tok::Colon && (next.tok == Tok::Comma || next.tok == close) {
                    p.bump();
                    out.push(Arg {
                        name: None,
                        value: Expr {
                            kind: ExprKind::Colon,
                            span: t.span,
                        },
                    });
                } else if t.tok == Tok::End {
                    return Err(
                        Diag::new("E-END-INDEX", "use `last` for the final index", t.span)
                            .label("`end` only closes blocks in Wisp")
                            .hint("write `last`, for example W[last, :]"),
                    );
                } else if matches!(t.tok, Tok::Ident(_)) && next.tok == Tok::Assign {
                    let (name, sp) = p.ident("a name")?;
                    if !allow_named {
                        return Err(Diag::new(
                            "E-PARSE",
                            "named arguments are only allowed inside [ ]",
                            sp,
                        )
                        .hint("pass options as string pairs, for example f(x, 'name', value)"));
                    }
                    p.bump(); // =
                    let v = p.expr()?;
                    out.push(Arg {
                        name: Some(name),
                        value: v,
                    });
                } else {
                    let v = p.expr()?;
                    if p.at(&Tok::Colon) {
                        return Err(slice_colon_error(p, v));
                    }
                    out.push(Arg {
                        name: None,
                        value: v,
                    });
                }
                if p.eat(&Tok::Comma) {
                    continue;
                }
                p.expect(close.clone(), &format!("`,` or {}", describe(&close)))?;
                break;
            }
            Ok(out)
        })
    }

    fn primary(&mut self) -> PResult<Expr> {
        let t = self.peek().clone();
        let sp = t.span;
        let kind = match t.tok {
            Tok::Num(v) => {
                self.bump();
                ExprKind::Num(v)
            }
            Tok::Str(s) => {
                self.bump();
                ExprKind::Str(s)
            }
            Tok::Ident(s) => {
                self.bump();
                ExprKind::Ident(s)
            }
            Tok::True => {
                self.bump();
                ExprKind::Bool(true)
            }
            Tok::False => {
                self.bump();
                ExprKind::Bool(false)
            }
            Tok::Last => {
                self.bump();
                if self.at(&Tok::LParen) && !self.peek().space_before {
                    ExprKind::Ident("last".into())
                } else {
                    ExprKind::Last
                }
            }
            Tok::LParen => {
                self.bump();
                let e = self.with_ctx(Ctx::Paren, |p| {
                    let e = p.expr()?;
                    p.expect(Tok::RParen, "`)`")?;
                    Ok(e)
                })?;
                let end = self.toks[self.pos - 1].span;
                return Ok(Expr {
                    kind: e.kind,
                    span: sp.to(end),
                });
            }
            Tok::LBracket => {
                self.bump();
                let rows = self.rows(Tok::RBracket)?;
                ExprKind::Matrix(rows)
            }
            Tok::LBrace => {
                self.bump();
                let rows = self.rows(Tok::RBrace)?;
                ExprKind::Cell(rows)
            }
            Tok::At => {
                self.bump();
                if self.at(&Tok::LParen) {
                    self.bump();
                    let mut params = Vec::new();
                    self.with_ctx(Ctx::Paren, |p| {
                        if !p.at(&Tok::RParen) {
                            loop {
                                let (n, _) = p.ident("a parameter name")?;
                                params.push(n);
                                if !p.eat(&Tok::Comma) {
                                    break;
                                }
                            }
                        }
                        p.expect(Tok::RParen, "`)`")?;
                        Ok(())
                    })?;
                    let body = self.with_ctx(Ctx::Stmt, |p| p.expr())?;
                    let span = sp.to(body.span);
                    return Ok(Expr {
                        kind: ExprKind::Lambda(Rc::new(params), Rc::new(body)),
                        span,
                    });
                }
                let (n, nsp) = self.ident("a function name or `(` after `@`")?;
                return Ok(Expr {
                    kind: ExprKind::FuncRef(n),
                    span: sp.to(nsp),
                });
            }
            Tok::End => {
                return Err(Diag::new("E-PARSE", "unexpected `end`", sp)
                    .hint("inside an index, use `last` for the final position"));
            }
            other => {
                return Err(Diag::new(
                    "E-PARSE",
                    format!("expected an expression, found {}", describe(&other)),
                    sp,
                ));
            }
        };
        Ok(Expr { kind, span: sp })
    }

    /// Rows of a matrix or cell literal, after the opening bracket.
    fn rows(&mut self, close: Tok) -> PResult<Vec<Vec<Expr>>> {
        self.with_ctx(Ctx::Matrix, |p| {
            let mut rows: Vec<Vec<Expr>> = Vec::new();
            let mut row: Vec<Expr> = Vec::new();
            loop {
                let t = p.peek().clone();
                if t.tok == close {
                    p.bump();
                    break;
                }
                match t.tok {
                    Tok::Semi | Tok::Newline => {
                        p.bump();
                        if !row.is_empty() {
                            rows.push(std::mem::take(&mut row));
                        }
                        continue;
                    }
                    Tok::Comma => {
                        p.bump();
                        continue;
                    }
                    Tok::Eof => {
                        return Err(Diag::new(
                            "E-PARSE",
                            format!("missing {} to close the literal", describe(&close)),
                            t.span,
                        ));
                    }
                    _ => {}
                }
                let e = p.expr()?;
                row.push(e);
            }
            if !row.is_empty() {
                rows.push(row);
            }
            Ok(rows)
        })
    }
}

fn slice_colon_error(p: &mut Parser, a: Expr) -> Diag {
    // We are at ':' after `a`. Look ahead for `b` and an optional `:s` to build a hint.
    let colon = p.peek().span;
    let toks: Vec<Tok> = (0..4)
        .map(|k| {
            p.toks
                .get(p.pos + k)
                .map(|t| t.tok.clone())
                .unwrap_or(Tok::Eof)
        })
        .collect();
    let mut span = a.span.to(colon);
    let mut hint = "write a..b, which includes both ends".to_string();
    if let (ExprKind::Num(av), Tok::Num(bv)) = (&a.kind, &toks[1]) {
        span = span.to(p.toks[p.pos + 1].span);
        if let (Tok::Colon, Tok::Num(sv)) = (&toks[2], &toks[3]) {
            span = span.to(p.toks[p.pos + 3].span);
            hint = format!(
                "Python's {av}:{bv}:{sv} stops before {bv}, so write {av}..{} by {sv}",
                bv - 1.0
            );
        } else {
            hint = format!(
                "Python's {av}:{bv} stops before {bv}, so write {av}..{}",
                bv - 1.0
            );
        }
    }
    Diag::new(
        "E-SLICE-COLON",
        "ranges use `..` and include both ends",
        span,
    )
    .label("this is a Python-style half-open slice")
    .hint(hint)
}

fn to_lvalue(e: Expr) -> PResult<LValue> {
    let span = e.span;
    let mut path = Vec::new();
    let mut cur = e;
    loop {
        match cur.kind {
            ExprKind::Ident(name) => {
                path.reverse();
                return Ok(LValue {
                    base: name,
                    path,
                    span,
                });
            }
            ExprKind::Postfix(inner, post) => {
                match &post {
                    Post::Call(_) => {
                        return Err(Diag::new(
                            "E-PAREN-INDEX",
                            "parentheses call functions; they cannot be assigned to",
                            cur.span,
                        )
                        .hint("to index, write x[...] with 0-based positions"));
                    }
                    Post::Transpose => {
                        return Err(Diag::new(
                            "E-PARSE",
                            "cannot assign to a transpose",
                            cur.span,
                        ));
                    }
                    _ => {}
                }
                path.push(post);
                cur = *inner;
            }
            _ => return Err(Diag::new("E-PARSE", "this cannot be assigned to", cur.span)),
        }
    }
}

pub fn describe(t: &Tok) -> String {
    match t {
        Tok::Num(v) => format!("number `{v}`"),
        Tok::Str(_) => "a string".into(),
        Tok::Ident(s) => format!("`{s}`"),
        Tok::Newline => "end of line".into(),
        Tok::Eof => "end of file".into(),
        Tok::RParen => "`)`".into(),
        Tok::RBracket => "`]`".into(),
        Tok::RBrace => "`}`".into(),
        other => format!("`{}`", tok_text(other)),
    }
}

fn tok_text(t: &Tok) -> &'static str {
    match t {
        Tok::Function => "function",
        Tok::End => "end",
        Tok::If => "if",
        Tok::Elseif => "elseif",
        Tok::Else => "else",
        Tok::For => "for",
        Tok::While => "while",
        Tok::Break => "break",
        Tok::Continue => "continue",
        Tok::Return => "return",
        Tok::By => "by",
        Tok::Last => "last",
        Tok::True => "true",
        Tok::False => "false",
        Tok::Plus => "+",
        Tok::Minus => "-",
        Tok::Star => "*",
        Tok::Slash => "/",
        Tok::Backslash => "\\",
        Tok::Caret => "^",
        Tok::DotStar => ".*",
        Tok::DotSlash => "./",
        Tok::DotCaret => ".^",
        Tok::Transpose => "'",
        Tok::DotTranspose => ".'",
        Tok::EqEq => "==",
        Tok::NotEq => "!=",
        Tok::Lt => "<",
        Tok::Le => "<=",
        Tok::Gt => ">",
        Tok::Ge => ">=",
        Tok::AndAnd => "&&",
        Tok::OrOr => "||",
        Tok::Bang => "!",
        Tok::Tilde => "~",
        Tok::Assign => "=",
        Tok::PlusEq => "+=",
        Tok::MinusEq => "-=",
        Tok::StarEq => "*=",
        Tok::SlashEq => "/=",
        Tok::DotDot => "..",
        Tok::Colon => ":",
        Tok::Comma => ",",
        Tok::Semi => ";",
        Tok::LParen => "(",
        Tok::LBracket => "[",
        Tok::LBrace => "{",
        Tok::Dot => ".",
        Tok::At => "@",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::parse;

    fn row_len(src: &str) -> usize {
        let p = parse(src).unwrap();
        match &p.body[0].kind {
            StmtKind::Assign(_, _, e) => match &e.kind {
                ExprKind::Matrix(rows) => rows[0].len(),
                _ => panic!("not a matrix"),
            },
            _ => panic!("not an assignment"),
        }
    }

    #[test]
    fn matrix_whitespace_rules() {
        assert_eq!(row_len("x = [1 -2]"), 2);
        assert_eq!(row_len("x = [1 - 2]"), 1);
        assert_eq!(row_len("x = [1-2]"), 1);
        assert_eq!(row_len("x = [a [1 2]]"), 2);
        assert_eq!(row_len("x = [a[0] 2]"), 2);
        assert_eq!(row_len("x = [0 90 -100 90 -100 -100]"), 6);
        assert_eq!(row_len("x = [f(a -1) 2]"), 2);
    }

    #[test]
    fn quote_rule() {
        let p = parse("y = u' * W\ns = 'u'").unwrap();
        assert_eq!(p.body.len(), 2);
    }

    #[test]
    fn errors_have_codes() {
        assert_eq!(
            parse("W1[540:600, 0] = 1").unwrap_err().code,
            "E-SLICE-COLON"
        );
        assert_eq!(parse("x = 1 % comment").unwrap_err().code, "E-LEX-PERCENT");
        assert_eq!(parse("x[end] = 1").unwrap_err().code, "E-END-INDEX");
        assert_eq!(parse("x(1) = 3").unwrap_err().code, "E-PAREN-INDEX");
    }

    #[test]
    fn multi_assign_and_functions() {
        let p = parse("[~, k] = max(s)\nfunction c = land(r, s)\n c = r + s;\nend").unwrap();
        assert_eq!(p.functions.len(), 1);
        assert!(matches!(p.body[0].kind, StmtKind::Multi(_, _)));
    }

    #[test]
    fn underscore_discards_outputs_without_changing_named_targets() {
        let p = parse("[_, _value, _, indices[0]] = outputs()").unwrap();
        let StmtKind::Multi(targets, _) = &p.body[0].kind else {
            panic!("expected multi-assignment");
        };
        assert_eq!(targets.len(), 4);
        assert!(targets[0].is_none());
        assert_eq!(targets[1].as_ref().unwrap().base, "_value");
        assert!(targets[2].is_none());
        assert_eq!(targets[3].as_ref().unwrap().base, "indices");
        assert!(parse("[_.field, x] = outputs()").is_err());
        assert!(parse("[_[0], x] = outputs()").is_err());
    }

    #[test]
    fn inline_if_and_ranges() {
        parse("if x < 0 || x > 9, dx = -dx; x = x + 2*dx; end").unwrap();
        parse("for k = 0..last(layers)\n y = k;\nend").unwrap();
        parse("W2[out=0..7, in=4] = [200, 200, -200, -200, 200, 200, -200, -200];").unwrap();
        parse("W1[0, 540..594 by 6] = 1").unwrap();
    }
}

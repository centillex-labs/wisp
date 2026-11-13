//! A deliberately restricted Python surface, validated with RustPython's parser.
//! This is defense in depth, not a general Python sandbox or memory limiter.
use ruff_python_ast::{
    self as ast,
    visitor::{self, Visitor},
    Expr, Stmt,
};

pub(crate) fn validate(source: &str) -> Result<(), String> {
    let parsed = ruff_python_parser::parse_module(source).map_err(|e| e.to_string())?;
    let mut policy = Policy {
        error: None,
        depth: 0,
        nodes: 0,
    };
    policy.visit_body(&parsed.syntax().body);
    policy.error.map_or(Ok(()), Err)
}

struct Policy {
    error: Option<String>,
    depth: usize,
    nodes: usize,
}
impl Policy {
    fn deny(&mut self, message: &str) {
        self.error
            .get_or_insert_with(|| format!("Python policy: {message}"));
    }
    fn enter(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes > 10_000 || self.depth >= 100 {
            self.deny("syntax complexity limit exceeded");
        }
        if self.error.is_some() {
            return false;
        }
        self.depth += 1;
        true
    }
    fn name(&mut self, name: &str) {
        if name.starts_with('_') && name != "_" {
            self.deny("private names are not available");
        }
    }
}
impl<'a> Visitor<'a> for Policy {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if !self.enter() {
            return;
        }
        match stmt {
            Stmt::FunctionDef(f) => {
                self.name(f.name.as_str());
                if f.is_async || !f.decorator_list.is_empty() || f.type_params.is_some() || f.returns.is_some() {
                    self.deny("async functions, decorators and annotations are not supported");
                }
            }
            Stmt::For(f) if !f.is_async => {}
            Stmt::Assign(_) | Stmt::AugAssign(_) | Stmt::While(_) | Stmt::If(_)
            | Stmt::Assert(_) | Stmt::Expr(_) | Stmt::Return(_) | Stmt::Break(_)
            | Stmt::Continue(_) | Stmt::Pass(_) => {}
            _ => self.deny("statement not allowed (imports, classes, exception handlers, context managers and dynamic scope are disabled)"),
        }
        if self.error.is_none() {
            visitor::walk_stmt(self, stmt);
        }
        self.depth -= 1;
    }
    fn visit_expr(&mut self, expr: &'a Expr) {
        if !self.enter() {
            return;
        }
        match expr {
            Expr::Name(n) => self.name(n.id.as_str()),
            Expr::Attribute(a) => {
                if !matches!(a.ctx, ast::ExprContext::Load)
                    || !matches!(a.attr.as_str(), "shape" | "values" | "transpose" | "matmul" | "add" | "T") {
                    self.deny("only public Wisp tensor attributes are available");
                }
            }
            Expr::BoolOp(_) | Expr::BinOp(_) | Expr::UnaryOp(_) | Expr::Lambda(_)
            | Expr::If(_) | Expr::Dict(_) | Expr::Set(_) | Expr::List(_) | Expr::Tuple(_)
            | Expr::ListComp(_) | Expr::SetComp(_) | Expr::DictComp(_)
            | Expr::Compare(_) | Expr::Call(_) | Expr::FString(_)
            | Expr::StringLiteral(_) | Expr::BytesLiteral(_) | Expr::NumberLiteral(_)
            | Expr::BooleanLiteral(_) | Expr::NoneLiteral(_) | Expr::EllipsisLiteral(_)
            | Expr::Subscript(_) | Expr::Slice(_) => {}
            _ => self.deny("expression not supported (generators, await and assignment expressions are disabled)"),
        }
        if self.error.is_none() {
            visitor::walk_expr(self, expr);
        }
        self.depth -= 1;
    }
    fn visit_parameter(&mut self, parameter: &'a ast::Parameter) {
        self.name(parameter.name.as_str());
        if parameter.annotation.is_some() {
            self.deny("annotations are disabled");
        }
        if self.error.is_none() {
            visitor::walk_parameter(self, parameter);
        }
    }
    fn visit_comprehension(&mut self, c: &'a ast::Comprehension) {
        if c.is_async {
            self.deny("async comprehensions are disabled");
        }
        if self.error.is_none() {
            visitor::walk_comprehension(self, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate;
    #[test]
    fn rejects_capability_and_control_bypasses() {
        for code in [
            "import os",
            "x.__class__",
            "__builtins__",
            "x.T = 1",
            "try:\n pass\nexcept:\n pass",
            "with x:\n pass",
            "class X:\n pass",
            "def f(__builtins__):\n pass",
            "x = (a for a in range(2))",
            "@x\ndef f():\n pass",
            "'{0.__class__}'.format(1)",
        ] {
            assert!(validate(code).is_err(), "allowed {code}");
        }
    }
    #[test]
    fn accepts_numeric_python() {
        validate("def twice(x):\n return x + x\na = [i * 2 for i in range(3)]\nfor i in range(3):\n a[i] += 1\nassert a == [1, 3, 5]\nb = tensor([1], 1, 1)\nc = b.T @ b").unwrap();
    }
}

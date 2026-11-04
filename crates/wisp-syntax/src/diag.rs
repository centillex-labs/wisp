//! Source spans and diagnostics with stable error codes.

use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
    pub fn to(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub code: &'static str,
    pub message: String,
    pub span: Span,
    pub label: Option<String>,
    pub hint: Option<String>,
}

impl Diag {
    pub fn new(code: &'static str, message: impl Into<String>, span: Span) -> Self {
        Diag {
            code,
            message: message.into(),
            span,
            label: None,
            hint: None,
        }
    }
    pub fn label(mut self, l: impl Into<String>) -> Self {
        self.label = Some(l.into());
        self
    }
    pub fn hint(mut self, h: impl Into<String>) -> Self {
        self.hint = Some(h.into());
        self
    }

    /// Render in the format given in the specification:
    ///
    /// ```text
    /// error[E-CODE]: message
    ///   --> file:line:col
    ///   |
    /// 3 | source line
    ///   |        ^^^^ label
    ///   = hint: ...
    /// ```
    pub fn render(&self, file: &str, src: &str) -> String {
        let (line_no, col, line) = locate(src, self.span.start);
        let width = (self.span.end.saturating_sub(self.span.start)).max(1);
        let width = width.min(line.len().saturating_sub(col - 1).max(1));
        let gutter = line_no.to_string().len();
        let pad = " ".repeat(gutter);
        let mut out = format!("error[{}]: {}\n", self.code, self.message);
        out += &format!("{pad}--> {file}:{line_no}:{col}\n");
        out += &format!("{pad} |\n");
        out += &format!("{line_no} | {line}\n");
        out += &format!("{pad} | {}{}", " ".repeat(col - 1), "^".repeat(width));
        if let Some(l) = &self.label {
            out += &format!(" {l}");
        }
        out += "\n";
        if let Some(h) = &self.hint {
            out += &format!("{pad} = hint: {h}\n");
        }
        out
    }
}

impl fmt::Display for Diag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.code, self.message)
    }
}

/// 1-based line and column (as people read them in an editor) of a byte offset.
pub fn locate(src: &str, offset: usize) -> (usize, usize, &str) {
    let offset = offset.min(src.len());
    let before = &src[..offset];
    let line_no = before.matches('\n').count() + 1;
    let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = src[line_start..]
        .find('\n')
        .map(|i| line_start + i)
        .unwrap_or(src.len());
    let col = src[line_start..offset].chars().count() + 1;
    (line_no, col, &src[line_start..line_end])
}

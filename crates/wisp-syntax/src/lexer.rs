//! Hand-written lexer. It records whether whitespace preceded each token,
//! which the parser needs for matrix literals, and applies the quote rule
//! (transpose vs. string) from the specification.

use crate::diag::{Diag, Span};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    // keywords
    Function,
    End,
    If,
    Elseif,
    Else,
    For,
    While,
    Break,
    Continue,
    Return,
    By,
    Last,
    True,
    False,
    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Backslash,
    Caret,
    DotStar,
    DotSlash,
    DotCaret,
    Transpose,
    DotTranspose,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    Tilde,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    DotDot,
    Colon,
    Comma,
    Semi,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Dot,
    At,
    Newline,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
    /// True when whitespace (or a comment) separates this token from the previous one.
    pub space_before: bool,
}

fn keyword(s: &str) -> Option<Tok> {
    Some(match s {
        "function" => Tok::Function,
        "end" => Tok::End,
        "if" => Tok::If,
        "elseif" => Tok::Elseif,
        "else" => Tok::Else,
        "for" => Tok::For,
        "while" => Tok::While,
        "break" => Tok::Break,
        "continue" => Tok::Continue,
        "return" => Tok::Return,
        "by" => Tok::By,
        "last" => Tok::Last,
        "true" => Tok::True,
        "false" => Tok::False,
        _ => return None,
    })
}

pub fn lex(src: &str) -> Result<Vec<Token>, Diag> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out: Vec<Token> = Vec::new();
    let mut space = true;
    while i < b.len() {
        let c = b[i];
        // whitespace
        if c == b' ' || c == b'\t' || c == b'\r' {
            i += 1;
            space = true;
            continue;
        }
        // comments
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            space = true;
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let start = i;
            i += 2;
            loop {
                if i + 1 >= b.len() {
                    return Err(Diag::new(
                        "E-LEX",
                        "unterminated block comment",
                        Span::new(start, start + 2),
                    )
                    .hint("close it with */"));
                }
                if b[i] == b'*' && b[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            space = true;
            continue;
        }
        // line continuation
        if c == b'.' && b.get(i + 1) == Some(&b'.') && b.get(i + 2) == Some(&b'.') {
            i += 3;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            if i < b.len() {
                i += 1;
            }
            space = true;
            continue;
        }
        if c == b'%' {
            return Err(Diag::new(
                "E-LEX-PERCENT",
                "`%` is not a comment in Wisp",
                Span::new(i, i + 1),
            )
            .label("MATLAB-style comment")
            .hint("use // for a line comment or /* ... */ for a block"));
        }
        let start = i;
        let push = |out: &mut Vec<Token>, tok: Tok, end: usize, space: bool| {
            out.push(Token {
                tok,
                span: Span::new(start, end),
                space_before: space,
            });
        };
        if c == b'\n' {
            i += 1;
            push(&mut out, Tok::Newline, i, space);
            space = true;
            continue;
        }
        // numbers
        if c.is_ascii_digit() {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let v: f64 = src[start..i]
                .parse()
                .map_err(|_| Diag::new("E-LEX", "bad number", Span::new(start, i)))?;
            push(&mut out, Tok::Num(v), i, space);
            space = false;
            continue;
        }
        // identifiers / keywords
        if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let s = &src[start..i];
            let tok = keyword(s).unwrap_or_else(|| Tok::Ident(s.to_string()));
            push(&mut out, tok, i, space);
            space = false;
            continue;
        }
        // quote: transpose or string
        if c == b'\'' {
            let prev_allows_transpose = !space
                && matches!(
                    out.last().map(|t| &t.tok),
                    Some(Tok::Ident(_))
                        | Some(Tok::Num(_))
                        | Some(Tok::RParen)
                        | Some(Tok::RBracket)
                        | Some(Tok::RBrace)
                        | Some(Tok::Transpose)
                        | Some(Tok::DotTranspose)
                        | Some(Tok::Last)
                );
            if prev_allows_transpose {
                i += 1;
                push(&mut out, Tok::Transpose, i, space);
                space = false;
                continue;
            }
        }
        if c == b'\'' || c == b'"' {
            let q = c;
            i += 1;
            let mut s = String::new();
            loop {
                if i >= b.len() || b[i] == b'\n' {
                    return Err(Diag::new(
                        "E-LEX",
                        "unterminated string",
                        Span::new(start, i),
                    ));
                }
                if b[i] == q {
                    if b.get(i + 1) == Some(&q) {
                        s.push(q as char);
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                // handle UTF-8 properly
                let ch = src[i..].chars().next().unwrap();
                s.push(ch);
                i += ch.len_utf8();
            }
            push(&mut out, Tok::Str(s), i, space);
            space = false;
            continue;
        }
        // operators (longest match first)
        let two = if i + 1 < b.len() { &src[i..i + 2] } else { "" };
        let tok2 = match two {
            ".." => Some(Tok::DotDot),
            ".*" => Some(Tok::DotStar),
            "./" => Some(Tok::DotSlash),
            ".^" => Some(Tok::DotCaret),
            ".'" => Some(Tok::DotTranspose),
            "==" => Some(Tok::EqEq),
            "!=" | "~=" => Some(Tok::NotEq),
            "<=" => Some(Tok::Le),
            ">=" => Some(Tok::Ge),
            "&&" => Some(Tok::AndAnd),
            "||" => Some(Tok::OrOr),
            "+=" => Some(Tok::PlusEq),
            "-=" => Some(Tok::MinusEq),
            "*=" => Some(Tok::StarEq),
            "/=" => Some(Tok::SlashEq),
            _ => None,
        };
        if let Some(t) = tok2 {
            i += 2;
            push(&mut out, t, i, space);
            space = false;
            continue;
        }
        let tok1 = match c {
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'\\' => Tok::Backslash,
            b'^' => Tok::Caret,
            b'<' => Tok::Lt,
            b'>' => Tok::Gt,
            b'!' => Tok::Bang,
            b'~' => Tok::Tilde,
            b'=' => Tok::Assign,
            b':' => Tok::Colon,
            b',' => Tok::Comma,
            b';' => Tok::Semi,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b'.' => Tok::Dot,
            b'@' => Tok::At,
            _ => {
                let ch = src[i..].chars().next().unwrap();
                return Err(Diag::new(
                    "E-LEX",
                    format!("unexpected character `{ch}`"),
                    Span::new(i, i + ch.len_utf8()),
                ));
            }
        };
        i += 1;
        push(&mut out, tok1, i, space);
        space = false;
    }
    out.push(Token {
        tok: Tok::Eof,
        span: Span::new(b.len(), b.len()),
        space_before: true,
    });
    Ok(out)
}

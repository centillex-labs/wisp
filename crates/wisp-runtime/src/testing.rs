//! Function-based tests. Discovery never executes the program body.

use crate::interp::{err, R};
use crate::value::{FuncVal, Value};
use crate::Interp;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use wisp_syntax::ast::{Function, Program};
use wisp_syntax::{Diag, Span};

#[derive(Default)]
pub struct TestContext {
    failures: Vec<Diag>,
    skipped: Option<String>,
}

#[derive(Clone, Copy)]
pub enum TestMethod {
    Printf,
    Error,
    Fail,
    Skip,
}

/// Bound methods retain their owning test when passed to helpers or aliased.
pub(crate) fn call_method(
    interpreter: &mut Interp,
    context: &RefCell<TestContext>,
    method: TestMethod,
    args: &[Value],
    span: Span,
) -> R<Vec<Value>> {
    let message = match method {
        TestMethod::Fail => {
            if !args.is_empty() {
                return Err(err("E-ARGS", "t.fail() takes no arguments", span));
            }
            "test marked failed".to_string()
        }
        TestMethod::Skip if args.is_empty() => "test skipped".to_string(),
        _ => {
            let Some(Value::Str(format)) = args.first() else {
                return Err(err("E-ARGS", "test method requires a format string", span));
            };
            crate::builtins::format(interpreter, format, &args[1..], span)?
        }
    };
    match method {
        TestMethod::Printf => interpreter.print(&message),
        TestMethod::Error | TestMethod::Fail => {
            context
                .borrow_mut()
                .failures
                .push(err("E-TEST-FAIL", message, span));
        }
        TestMethod::Skip => {
            context.borrow_mut().skipped = Some(message.clone());
            return Err(err("E-TEST-SKIP", message, span));
        }
    }
    Ok(vec![])
}

pub struct TestResult {
    pub name: String,
    pub output: String,
    pub failures: Vec<Diag>,
    pub skipped: Option<String>,
    pub observations: u64,
}

/// Validate the whole suite before running any test, in declaration order.
/// Duplicate helper names are rejected too: loading them would silently shadow
/// functions that a test might call.
pub fn discover(program: &Program) -> R<Vec<Rc<Function>>> {
    let mut names = HashSet::new();
    let mut tests = Vec::new();
    for function in &program.functions {
        if !names.insert(&function.name) {
            return Err(err(
                "E-TEST",
                format!("duplicate function `{}`", function.name),
                function.span,
            ));
        }
        if function.name.starts_with("test_") {
            if function.params.len() != 1 || !function.outs.is_empty() {
                return Err(err(
                    "E-TEST",
                    format!(
                        "test `{}` must take one context argument and declare no outputs",
                        function.name
                    ),
                    function.span,
                ));
            }
            tests.push(function.clone());
        }
    }
    Ok(tests)
}

/// Each call owns a fresh interpreter, harness, function locals, and timer.
/// Filesystem/network side effects are not isolated. Top-level statements are
/// deliberately not evaluated; setup belongs in the test or a helper function.
pub fn run(
    file: &str,
    source: &str,
    program: &Program,
    test: &Function,
    budget: Option<u64>,
) -> TestResult {
    let mut interpreter = Interp::new(file, source);
    interpreter.echo = false;
    interpreter.harness.passes = budget;
    interpreter.load_functions(program);
    let state = Rc::new(RefCell::new(TestContext::default()));
    let mut fields = [
        ("name".into(), Value::str(&test.name)),
        ("file".into(), Value::str(file)),
    ]
    .into_iter()
    .collect::<indexmap::IndexMap<_, _>>();
    for (name, method) in [
        ("fprintf", TestMethod::Printf),
        ("error", TestMethod::Error),
        ("fail", TestMethod::Fail),
        ("skip", TestMethod::Skip),
    ] {
        fields.insert(
            name.into(),
            Value::Func(Rc::new(FuncVal::TestMethod {
                context: state.clone(),
                method,
            })),
        );
    }
    let result = interpreter.call_named(&test.name, vec![Value::strukt(fields)], 0, test.span);
    let mut state = state.borrow_mut();
    if let Err(error) = result {
        if error.code != "E-TEST-SKIP" || state.skipped.is_none() {
            state.failures.push(error);
        }
    }
    TestResult {
        name: test.name.clone(),
        output: interpreter.out,
        failures: std::mem::take(&mut state.failures),
        skipped: state.skipped.take(),
        observations: interpreter.harness.used,
    }
}

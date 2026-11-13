//! Self-contained, in-process RustPython frontend. No external Python executable.
//! Restricted capabilities are not hard process/memory isolation: see docs/python-security.md.
mod policy;
mod tensors;

use rustpython_vm::{
    builtins::{PyBaseExceptionRef, PyFloat, PyInt, PyList, PyStr, PyTuple},
    compiler::Mode,
    function::FuncArgs,
    scope::Scope,
    Interpreter, PyObjectRef, PyResult, VirtualMachine,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tensors::{Argument, Output, Tensors, FUNCTIONS, MAX_ELEMENTS};

pub const MAX_SOURCE_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_FEEDS: usize = 64;
const MAX_CALLS: usize = 1000;
const MAX_TRACE_EVENTS: usize = 100_000;
const EXECUTION_LIMIT: Duration = Duration::from_secs(2);

pub struct Execution {
    pub output: String,
}

#[derive(Default)]
struct Host {
    tensors: Tensors,
    calls: usize,
    output_bytes: usize,
    output: String,
    trace_events: usize,
    started: Option<Instant>,
    exhausted: bool,
}
struct Embedded {
    // Drop Python references before dropping their interpreter.
    scope: Scope,
    interpreter: Interpreter,
}

/// A persistent embedded interpreter. Private host quotas survive all submissions.
/// All failed submissions close the session. This is not a hostile-code OS sandbox.
pub struct Session {
    embedded: Option<Embedded>,
    host: Arc<Mutex<Host>>,
    feeds: usize,
}

/// Run a session synchronously on an in-process thread with enough native stack
/// for RustPython's bootstrap (including debug builds). This is NOT a process,
/// RPC boundary, or crash sandbox. Tensor callbacks still execute directly.
pub fn with_session<R: Send + 'static>(
    run: impl FnOnce(&mut Session) -> Result<R, String> + Send + 'static,
) -> Result<R, String> {
    std::thread::Builder::new()
        .name("wisp-python".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut session = Session::new()?;
            run(&mut session)
        })
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "embedded Python interpreter panicked".to_owned())?
}

impl Session {
    /// Create an interpreter on the calling thread. The caller must provide a
    /// sufficiently large native stack (16 MiB recommended for debug builds).
    /// `with_session` provides that stack automatically, in the same process.
    pub fn new() -> Result<Self, String> {
        let mut settings = rustpython_vm::Settings::default();
        settings.isolated = true;
        settings.import_site = false;
        settings.install_signal_handlers = false;
        let interpreter = Interpreter::without_stdlib(settings);
        let host = Arc::new(Mutex::new(Host::default()));
        let scope = interpreter.enter(|vm| -> Result<Scope, String> {
            vm.recursion_limit.set(100);
            let globals = vm.ctx.new_dict();
            let builtins = vm.ctx.new_dict();
            // Copy a small pure builtin surface. No import, open, eval, exec,
            // compile, getattr, globals, locals, vars, input, or introspection helpers.
            for name in [
                "abs",
                "all",
                "any",
                "bool",
                "dict",
                "enumerate",
                "float",
                "int",
                "len",
                "list",
                "max",
                "min",
                "range",
                "reversed",
                "round",
                "set",
                "sorted",
                "str",
                "sum",
                "tuple",
                "zip",
                "__build_class__",
                "property",
                "object",
            ] {
                let value = vm
                    .builtins
                    .get_attr(name, vm)
                    .map_err(|e| exception(vm, &e))?;
                builtins
                    .set_item(name, value, vm)
                    .map_err(|e| exception(vm, &e))?;
            }
            globals
                .set_item("__builtins__", builtins.clone().into(), vm)
                .map_err(|e| exception(vm, &e))?;
            globals
                .set_item("__name__", vm.ctx.new_str("wisp").into(), vm)
                .map_err(|e| exception(vm, &e))?;
            let scope = Scope::with_builtins(None, globals, vm);
            for &name in FUNCTIONS {
                let state = host.clone();
                let function = vm.new_function(
                    name,
                    move |args: FuncArgs, vm: &VirtualMachine| -> PyResult {
                        // Convert primitives before taking the host lock. Never invoke user
                        // conversion methods while a host mutex is held.
                        let args = convert_args(name, args, vm)?;
                        let mut state = state.lock().unwrap();
                        charge(&mut state, vm)?;
                        let result = state
                            .tensors
                            .call(name, &args)
                            .map_err(|e| vm.new_value_error(e))?;
                        Ok(to_python(result, vm))
                    },
                );
                scope
                    .globals
                    .set_item(name, function.into(), vm)
                    .map_err(|e| exception(vm, &e))?;
            }
            let state = host.clone();
            let print = vm.new_function(
                "print",
                move |args: FuncArgs, vm: &VirtualMachine| -> PyResult {
                    let sep = string_keyword(&args, "sep", " ", vm)?;
                    let end = string_keyword(&args, "end", "\n", vm)?;
                    if args
                        .kwargs
                        .keys()
                        .any(|key| key.as_bytes() != b"sep" && key.as_bytes() != b"end")
                    {
                        return Err(vm.new_type_error("print accepts only sep and end keywords"));
                    }
                    let mut text = String::new();
                    for (i, arg) in args.args.iter().enumerate() {
                        if i != 0 {
                            append(&mut text, &sep).map_err(|e| vm.new_runtime_error(e))?;
                        }
                        let rendered = arg.str(vm)?;
                        append(&mut text, &rendered.to_string_lossy())
                            .map_err(|e| vm.new_runtime_error(e))?;
                    }
                    append(&mut text, &end).map_err(|e| vm.new_runtime_error(e))?;
                    let mut state = state.lock().unwrap();
                    charge(&mut state, vm)?;
                    if text.len() > MAX_OUTPUT_BYTES - state.output_bytes {
                        state.exhausted = true;
                        return Err(vm.new_runtime_error("session output limit exceeded"));
                    }
                    state.output_bytes += text.len();
                    state.output.push_str(&text);
                    Ok(vm.ctx.none())
                },
            );
            builtins
                .set_item("print", print.into(), vm)
                .map_err(|e| exception(vm, &e))?;
            // Trusted wrapper definitions are embedded in the executable. They share
            // this same builtin dictionary, not the unrestricted VM builtins.
            vm.run_string(scope.clone(), include_str!("prelude.py"), "<wisp-prelude>")
                .map_err(|e| exception(vm, &e))?;
            for name in ["__build_class__", "property", "object"] {
                builtins.del_item(name, vm).map_err(|e| exception(vm, &e))?;
            }
            let state = host.clone();
            let trace = vm.new_function(
                "wisp_trace",
                move |args: FuncArgs, vm: &VirtualMachine| -> PyResult {
                    {
                        let mut state = state.lock().unwrap();
                        state.trace_events += 1;
                        if state.exhausted
                            || state.trace_events > MAX_TRACE_EVENTS
                            || state.started.is_some_and(|t| t.elapsed() > EXECUTION_LIMIT)
                        {
                            state.exhausted = true;
                            return Err(vm.new_runtime_error("Python execution limit exceeded"));
                        }
                    }
                    if let Some(frame) = args.args.first() {
                        frame.set_attr("f_trace_opcodes", vm.ctx.new_bool(true), vm)?;
                    }
                    Ok(vm.trace_func.borrow().clone())
                },
            );
            vm.trace_func.replace(trace.into());
            Ok(scope)
        })?;
        Ok(Self {
            embedded: Some(Embedded { scope, interpreter }),
            host,
            feeds: 0,
        })
    }

    pub fn execute(&mut self, source: &str) -> Result<Execution, String> {
        if self.embedded.is_none() {
            return Err("Python session is closed".into());
        }
        let validation = if source.len() > MAX_SOURCE_BYTES || self.feeds >= MAX_FEEDS {
            Err("Python source/session feed limit exceeded".into())
        } else {
            policy::validate(source)
        };
        if let Err(error) = validation {
            self.close();
            return Err(error);
        }
        self.feeds += 1;
        {
            let mut host = self.host.lock().unwrap();
            host.output.clear();
            host.trace_events = 0;
            host.started = Some(Instant::now());
        }
        let embedded = self.embedded.as_ref().unwrap();
        let result = embedded.interpreter.enter(|vm| {
            let code = vm
                .compile(source, Mode::Exec, "<wisp-python>".to_owned())
                .map_err(|e| e.to_string())?;
            vm.use_tracing.set(true);
            let result = vm.run_code_obj(code, embedded.scope.clone());
            vm.use_tracing.set(false);
            result.map(|_| ()).map_err(|e| exception(vm, &e))
        });
        let result = if self.host.lock().unwrap().exhausted {
            Err("Python session resource limit exceeded".into())
        } else {
            result
        };
        match result {
            Ok(()) => Ok(Execution {
                output: std::mem::take(&mut self.host.lock().unwrap().output),
            }),
            Err(e) => {
                self.close();
                Err(e)
            }
        }
    }

    pub fn close(&mut self) {
        self.embedded.take();
        // Release retained tensors/output as well as the interpreter.
        *self.host.lock().unwrap() = Host::default();
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.close();
    }
}

fn charge(host: &mut Host, vm: &VirtualMachine) -> PyResult<()> {
    if host.exhausted || host.calls >= MAX_CALLS {
        host.exhausted = true;
        return Err(vm.new_runtime_error("session host-call limit exceeded"));
    }
    host.calls += 1;
    Ok(())
}
fn append(output: &mut String, text: &str) -> Result<(), &'static str> {
    if text.len() > MAX_OUTPUT_BYTES - output.len() {
        return Err("output limit exceeded");
    }
    output.push_str(text);
    Ok(())
}
fn string_keyword(
    args: &FuncArgs,
    name: &str,
    default: &str,
    vm: &VirtualMachine,
) -> PyResult<String> {
    match args.kwargs.get(name) {
        None => Ok(default.to_owned()),
        Some(value) if vm.is_none(value) => Ok(default.to_owned()),
        Some(value) => value
            .downcast_ref::<PyStr>()
            .map(|s| s.to_string_lossy().into_owned())
            .ok_or_else(|| vm.new_type_error("print separator/end must be strings")),
    }
}
fn convert_args(name: &str, args: FuncArgs, vm: &VirtualMachine) -> PyResult<Vec<Argument>> {
    if !args.kwargs.is_empty() {
        return Err(vm.new_type_error("tensor functions accept positional arguments only"));
    }
    if args.args.len() > 3 {
        return Err(vm.new_type_error("too many tensor arguments"));
    }
    args.args
        .iter()
        .enumerate()
        .map(|(i, value)| {
            if name == "wisp_tensor" && i == 0 {
                let data = if let Some(list) = value.downcast_ref::<PyList>() {
                    numbers(&list.borrow_vec(), vm)?
                } else if let Some(tuple) = value.downcast_ref::<PyTuple>() {
                    numbers(tuple.as_slice(), vm)?
                } else {
                    return Err(vm.new_type_error("tensor data must be a list or tuple"));
                };
                Ok(Argument::Data(data))
            } else {
                let integer = value
                    .downcast_ref::<PyInt>()
                    .ok_or_else(|| vm.new_type_error("expected a non-negative integer"))?;
                Ok(Argument::Integer(integer.try_to_primitive::<usize>(vm)?))
            }
        })
        .collect()
}
fn numbers(values: &[PyObjectRef], vm: &VirtualMachine) -> PyResult<Vec<f64>> {
    if values.len() > MAX_ELEMENTS {
        return Err(vm.new_value_error("tensor size limit exceeded"));
    }
    values
        .iter()
        .map(|value| {
            if let Some(f) = value.downcast_ref::<PyFloat>() {
                Ok(f.to_f64())
            } else if let Some(i) = value.downcast_ref::<PyInt>() {
                Ok(i.try_to_primitive::<i64>(vm)? as f64)
            } else {
                Err(vm.new_type_error("tensor data must contain plain numbers"))
            }
        })
        .collect()
}
fn to_python(result: Output, vm: &VirtualMachine) -> PyObjectRef {
    match result {
        Output::Handle(id) => vm.ctx.new_int(id).into(),
        Output::Shape(shape) => vm
            .ctx
            .new_tuple(
                shape
                    .into_iter()
                    .map(|n| vm.ctx.new_int(n).into())
                    .collect(),
            )
            .into(),
        Output::Data(data) => vm
            .ctx
            .new_list(
                data.into_iter()
                    .map(|n| vm.ctx.new_float(n).into())
                    .collect(),
            )
            .into(),
    }
}
struct Diagnostic(String);
impl rustpython_vm::py_io::Write for Diagnostic {
    type Error = ();
    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> Result<(), ()> {
        std::fmt::Write::write_fmt(self, args).map_err(|_| ())
    }
}
impl std::fmt::Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        append(&mut self.0, text).map_err(|_| std::fmt::Error)
    }
}
fn exception(vm: &VirtualMachine, e: &PyBaseExceptionRef) -> String {
    let mut diagnostic = Diagnostic(String::new());
    let _ = vm.write_exception(&mut diagnostic, e);
    diagnostic.0
}

//! `wisp`: execute programs and validate syntax.

use std::io::{BufRead, Write as IoWrite};
use std::process::ExitCode;
use std::rc::Rc;
use wisp_runtime::value::Value;
use wisp_runtime::Interp;

const USAGE: &str = "\
wisp: a language for Weight-Space Programming

usage:
  wisp <program.wisp> [--budget N] [--log FILE]
  wisp <program.wisp> --check
  wisp test <program.wisp> [--budget N]
  wisp kernel <setup.wisp> [--log FILE]
  wisp python <script.py>
  wisp python-kernel

options:
  --check       validate syntax without executing the program
  --budget N    limit model-observation passes
  --log FILE    write observation records as JSONL
  --help, -h    show help
";

pub mod executable;
#[cfg(feature = "python")]
mod python;

pub fn main_cli() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(|s| s.as_str()) {
        Some("--help" | "-h") => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        #[cfg(feature = "python")]
        Some("python" | "python-kernel") => python::run(&args),
        #[cfg(not(feature = "python"))]
        Some("python" | "python-kernel") => Err(
            "Python frontend requires a build with --features python; see docs/python-security.md"
                .into(),
        ),
        Some("test") => cmd_test(&args[1..]),
        Some("kernel") => cmd_kernel(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("jit") => cmd_jit(&args[1..]),
        Some("check") => cmd_check(&args[1..]),
        Some("emit-llvm") => cmd_emit(&args[1..]),
        Some("apply") => cmd_apply(&args[1..]),
        Some("edits2w") => cmd_edits2w(&args[1..]),
        #[cfg(feature = "python")]
        Some(path) if path.ends_with(".py") => {
            let mut python_args = vec!["python".to_owned()];
            python_args.extend_from_slice(&args);
            python::run(&python_args)
        }
        Some(_) => cmd_program(&args),
        None => {
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprint!("{msg}");
            if !msg.ends_with('\n') {
                eprintln!();
            }
            ExitCode::from(1)
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

fn cmd_program(args: &[String]) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--task") {
        return Err("--task is not supported here; define budget(), dataset(), and protect() in the program".into());
    }
    if args.iter().any(|arg| arg == "--check") {
        let path = match args {
            [flag, path] if flag == "--check" && !path.starts_with('-') => path,
            [path, flag] if flag == "--check" && !path.starts_with('-') => path,
            _ => return Err("usage: wisp <program.wisp> --check (execution options cannot be combined with --check)".into()),
        };
        return cmd_check(std::slice::from_ref(path));
    }
    if args.first().is_none_or(|arg| arg.starts_with('-')) {
        return Err(USAGE.into());
    }
    cmd_run(args)
}

fn cmd_test(args: &[String]) -> Result<(), String> {
    const HELP: &str = "usage: wisp test <program.wisp> [--budget N]\n\nRuns test_* functions in declaration order with a fresh context per test.\nTop-level statements are not executed. --budget is a per-test limit.\n";
    if matches!(args, [flag] if flag == "--help" || flag == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let (path, budget) = match args {
        [path] if !path.starts_with('-') => (path, None),
        [path, flag, value] if !path.starts_with('-') && flag == "--budget" => {
            let budget = value
                .parse::<u64>()
                .map_err(|_| "--budget needs a nonnegative integer")?;
            (path, Some(budget))
        }
        _ => return Err(HELP.into()),
    };
    let src = read(path)?;
    let program = wisp_syntax::parse(&src).map_err(|d| d.render(path, &src))?;
    let tests = wisp_runtime::testing::discover(&program).map_err(|d| d.render(path, &src))?;
    if tests.is_empty() {
        return Err(format!("{path}: no test_* functions found"));
    }
    let mut failed = 0;
    let mut skipped = 0;
    for test in &tests {
        let result = wisp_runtime::testing::run(path, &src, &program, test, budget);
        print!("{}", result.output);
        if !result.output.is_empty() && !result.output.ends_with('\n') {
            println!();
        }
        if !result.failures.is_empty() {
            failed += 1;
            println!("FAIL {}", result.name);
            for error in result.failures {
                eprint!("{}", error.render(path, &src));
            }
        } else if let Some(reason) = result.skipped {
            skipped += 1;
            println!("SKIP {}: {reason}", result.name);
        } else {
            println!("PASS {}", result.name);
        }
    }
    let skip_summary = if skipped > 0 {
        format!(", {skipped} skipped")
    } else {
        String::new()
    };
    println!(
        "{} passed, {failed} failed{skip_summary}, {} total",
        tests.len() - failed - skipped,
        tests.len()
    );
    if failed > 0 {
        Err(format!("{path}: test suite failed"))
    } else {
        Ok(())
    }
}

/// Persistent statement session: one setup program, one environment, one budget.
/// Each submitted chunk is answered with a single JSON line; a blank line ends a
/// chunk and `exit` or end of input ends the session. Parse and runtime errors
/// are reported and do not end the session.
fn cmd_kernel(args: &[String]) -> Result<(), String> {
    const HELP: &str = "usage: wisp kernel <setup.wisp> [--log FILE] [--transcript FILE] [--resume FILE]\n\nRuns the setup program once, then evaluates statements from stdin against the\nsame interpreter, environment, and observation budget. End each chunk with a\nblank line; send `exit` or end of input to stop. One JSON line per chunk.\n--transcript records every chunk and response as JSON lines. --resume replays\na transcript's chunks against a fresh session before reading stdin; chunks are\nre-executed, so replay requires deterministic programs.\n";
    if matches!(args, [flag] if flag == "--help" || flag == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let (path, log, transcript, resume) = {
        let mut values = (None, None, None, None);
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--log" => values.1 = args.get(i + 1).cloned(),
                "--transcript" => values.2 = args.get(i + 1).cloned(),
                "--resume" => values.3 = args.get(i + 1).cloned(),
                other if !other.starts_with('-') && values.0.is_none() => {
                    values.0 = Some(other.to_string())
                }
                _ => return Err(HELP.into()),
            }
            if matches!(args[i].as_str(), "--log" | "--transcript" | "--resume") {
                if args.get(i + 1).is_none() {
                    return Err(HELP.into());
                }
                i += 2;
            } else {
                i += 1;
            }
        }
        if values.0.is_none() {
            return Err(HELP.into());
        }
        values
    };
    let path = path.as_deref().ok_or_else(|| HELP.to_string())?;
    let src = read(path)?;
    let program = wisp_syntax::parse(&src).map_err(|d| d.render(path, &src))?;
    let mut it = Interp::new(path, &src);
    it.echo = false;
    if let Some(l) = log {
        let _ = std::fs::remove_file(&l);
        it.harness.log_path = Some(l);
    }
    let mut env = wisp_runtime::interp::Env::new();
    it.eval_program(&mut env, &program)
        .map_err(|d| d.render(path, &src))?;
    let mut transcript = transcript
        .map(|file| {
            let new = !std::path::Path::new(&file).exists();
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file)
                .map_err(|e| format!("{file}: {e}"))?;
            if new {
                let record = serde_json::json!({"event": "setup", "source": src});
                writeln!(&f, "{record}").map_err(|e| format!("{file}: {e}"))?;
            }
            Ok::<std::fs::File, String>(f)
        })
        .transpose()?;
    let status = |it: &mut Interp, ok: bool, extra: serde_json::Value| {
        let remaining = it.harness.passes.map(|p| p.saturating_sub(it.harness.used));
        let mut value = serde_json::json!({
            "ok": ok,
            "budget_used": it.harness.used,
            "budget_remaining": remaining,
            "output": it.out,
        });
        if let Some(map) = value.as_object_mut() {
            if let serde_json::Value::Object(fields) = extra {
                for (k, v) in fields {
                    if k != "output" {
                        map.insert(k, v);
                    }
                }
            }
        }
        println!("{value}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    };
    status(&mut it, true, serde_json::json!({"event": "ready"}));
    let replay = resume
        .map(|file| -> Result<Vec<String>, String> {
            let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
            let mut chunks = Vec::new();
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let record: serde_json::Value =
                    serde_json::from_str(line).map_err(|e| format!("{file}: {e}"))?;
                match record.get("event").and_then(|e| e.as_str()) {
                    Some("setup") => {
                        if record.get("source").and_then(|s| s.as_str()) != Some(src.as_str()) {
                            return Err(format!(
                            "{file}: the transcript was recorded against a different setup program"
                        ));
                        }
                    }
                    Some("chunk") => {
                        if let Some(text) = record.get("chunk").and_then(|c| c.as_str()) {
                            chunks.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
            Ok(chunks)
        })
        .transpose()?;
    let mut evaluate = |it: &mut Interp, snippet: &str| -> serde_json::Value {
        it.out.clear();
        let mut value = match wisp_syntax::parse(snippet) {
            Err(d) => serde_json::json!({
                "ok": false,
                "error": d.render("<stdin>", snippet),
            }),
            Ok(program) => match it.eval_program(&mut env, &program) {
                Ok(()) => serde_json::json!({"ok": true}),
                Err(d) => serde_json::json!({
                    "ok": false,
                    "error": d.render("<stdin>", snippet),
                }),
            },
        };
        let remaining = it.harness.passes.map(|p| p.saturating_sub(it.harness.used));
        let map = value.as_object_mut().unwrap();
        map.insert("budget_used".into(), it.harness.used.into());
        map.insert("budget_remaining".into(), remaining.into());
        map.insert("output".into(), it.out.clone().into());
        value
    };
    if let Some(chunks) = replay {
        for snippet in chunks {
            let response = evaluate(&mut it, &snippet);
            if response["ok"].as_bool() != Some(true) {
                return Err(format!(
                    "--resume: a replayed chunk failed:\n{}",
                    response["error"].as_str().unwrap_or_default()
                ));
            }
        }
    }
    let mut chunk = String::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| format!("stdin: {e}"))?;
        if line.trim().is_empty() {
            if chunk.is_empty() {
                continue;
            }
        } else if line.trim() == "exit" {
            // Flush any pending chunk so a final statement without a trailing
            // blank line is not silently discarded.
            if !chunk.is_empty() {
                let text = std::mem::take(&mut chunk);
                let response = evaluate(&mut it, text.as_str());
                if let Some(f) = transcript.as_mut() {
                    let record =
                        serde_json::json!({"event": "chunk", "chunk": text, "response": response});
                    let _ = writeln!(f, "{record}");
                }
                let ok = response["ok"].as_bool() == Some(true);
                status(&mut it, ok, response);
            }
            break;
        }
        if line.trim().is_empty() {
            if chunk.is_empty() {
                continue;
            }
            if chunk.trim() == "exit" {
                break;
            }
            let text = std::mem::take(&mut chunk);
            let snippet = text.as_str();
            let response = evaluate(&mut it, snippet);
            if let Some(f) = transcript.as_mut() {
                let record =
                    serde_json::json!({"event": "chunk", "chunk": text, "response": response});
                if let Err(e) = writeln!(f, "{record}") {
                    return Err(format!("{e}"));
                }
            }
            let ok = response["ok"].as_bool() == Some(true);
            status(&mut it, ok, response);
        } else {
            chunk.push_str(&line);
            chunk.push('\n');
        }
    }
    status(&mut it, true, serde_json::json!({"event": "exit"}));
    Ok(())
}

fn cmd_check(args: &[String]) -> Result<(), String> {
    if args.len() != 1 || args[0].starts_with('-') {
        return Err("usage: wisp <program.wisp> --check".into());
    }
    let path = &args[0];
    let src = read(path)?;
    match wisp_syntax::parse(&src) {
        Ok(p) => {
            println!(
                "{path}: OK ({} statements, {} functions)",
                p.body.len(),
                p.functions.len()
            );
            Ok(())
        }
        Err(d) => Err(d.render(path, &src)),
    }
}

fn cmd_emit(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("emit-llvm needs a file")?;
    if args.len() > 2 || (args.len() == 2 && args[1] != "--native") {
        return Err("usage: wisp emit-llvm <file.wisp> [--native]".into());
    }
    let src = read(path)?;
    let prog = wisp_syntax::parse(&src).map_err(|d| d.render(path, &src))?;
    if args.len() == 2 {
        print!(
            "{}",
            wisp_llvm::ir::native::program(&prog).map_err(|d| d.render(path, &src))?
        );
        return Ok(());
    }
    let fs: Vec<&wisp_syntax::ast::Function> = prog.functions.iter().map(|f| f.as_ref()).collect();
    let m = wisp_llvm::ir::scalar_module(&fs);
    print!("{}", m.ir);
    for (n, why) in &m.rejected {
        eprintln!("; not compiled: {n}: {why}");
    }
    Ok(())
}

fn cmd_apply(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("apply needs <base> <patch.json> <out>".into());
    }
    let log = wisp_runtime::io::apply_patch(&args[0], &args[1], &args[2])?;
    print!("{log}");
    println!("wrote {}", args[2]);
    Ok(())
}

fn configure_task(it: &mut Interp, task_path: &str) -> Result<(), String> {
    let src = read(task_path)?;
    let prog = wisp_syntax::parse(&src).map_err(|d| d.render(task_path, &src))?;
    let mut t = Interp::new(task_path, &src);
    t.echo = false;
    let env = t.run(&prog).map_err(|d| d.render(task_path, &src))?;
    let task = match env.get("task") {
        Some(Value::Struct(s)) => s.clone(),
        _ => {
            return Err(format!(
                "{task_path}: a task file must set the struct `task`"
            ))
        }
    };
    let strings = |v: Option<&Value>| -> Vec<String> {
        match v {
            Some(Value::Cell(c)) => c
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect(),
            Some(Value::Str(s)) => vec![s.to_string()],
            _ => vec![],
        }
    };
    if let Some(Value::Struct(b)) = task.get("budget") {
        if let Some(Value::Arr(p)) = b.get("passes") {
            it.harness.passes = Some(p.data[0] as u64);
        }
    }
    it.harness.allow.extend(strings(task.get("allow")));
    it.harness.protect.extend(strings(task.get("protect")));
    it.harness.test_sets.extend(strings(task.get("test")));
    it.harness
        .practice_sets
        .extend(strings(task.get("practice")));
    if let Some(data) = task.get("activations") {
        let Value::Struct(data) = data else {
            return Err("task.activations must be a struct of numeric arrays".into());
        };
        for (name, value) in data.iter() {
            it.harness
                .register_activations(name, value.clone())
                .map_err(|m| format!("{task_path}: {m}"))?;
        }
    }
    if let Some(data) = task.get("datasets") {
        let Value::Struct(data) = data else {
            return Err("task.datasets must be a struct of named arrays or cells".into());
        };
        for (name, value) in data.iter() {
            it.harness
                .register_dataset(name, value.clone(), false)
                .map_err(|m| format!("{task_path}: {m}"))?;
        }
    }
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("run needs a file")?;
    let src = read(path)?;
    run_interpreted(path, &src, &args[1..])
}

fn cmd_jit(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: wisp jit <file.wisp>".into());
    }
    let src = read(&args[0])?;
    run_native(&args[0], &src)
}

/// No Interp is constructed on this path: compilation covers the entire script.
pub fn run_native(path: &str, src: &str) -> Result<(), String> {
    let prog = wisp_syntax::parse(src).map_err(|d| d.render(path, src))?;
    let ir = wisp_llvm::ir::native::program(&prog).map_err(|d| d.render(path, src))?;
    let lib = wisp_llvm::jit::compile_ir(&ir, "-O2")?;
    let mut native = wisp_llvm::jit::NativeProgram::load(lib)?;
    native.run().map_err(|d| d.render(path, src))
}

/// Shared execution entry point for scripts and packaged executables.
pub fn run_source(path: &str, src: &str, args: &[String]) -> Result<(), String> {
    if let Some(i) = args.iter().position(|a| a == "--interpret") {
        let mut options = args.to_vec();
        options.remove(i);
        return run_interpreted(path, src, &options);
    }
    if !args.is_empty() {
        return Err(format!("native execution is the default and accepts no runtime flags (got `{}`); use --interpret explicitly for interpreter/harness options", args[0]));
    }
    run_native(path, src)
}

fn run_interpreted(path: &str, src: &str, args: &[String]) -> Result<(), String> {
    validate_run_args(args)?;
    let prog = wisp_syntax::parse(src).map_err(|d| d.render(path, src))?;
    let mut it = Interp::new(path, src);
    if let Some(t) = flag(args, "--task") {
        configure_task(&mut it, &t)?;
    }
    if let Some(b) = flag(args, "--budget") {
        it.harness.passes = Some(b.parse().map_err(|_| "--budget needs a number")?);
    }
    if let Some(a) = flag(args, "--allow") {
        it.harness.allow.push(a);
    }
    if let Some(l) = flag(args, "--log") {
        let _ = std::fs::remove_file(&l);
        it.harness.log_path = Some(l);
    }
    if args.iter().any(|a| a == "--jit") {
        let fs: Vec<&wisp_syntax::ast::Function> =
            prog.functions.iter().map(|f| f.as_ref()).collect();
        let m = wisp_llvm::ir::scalar_module(&fs);
        if !m.compiled.is_empty() {
            let lib = wisp_llvm::jit::compile_ir(&m.ir, "-O2")?;
            it.native = Some(wisp_llvm::jit::NativeScalar::load(lib, &m.compiled)?);
        }
        let names: Vec<&str> = m.compiled.iter().map(|c| c.0.as_str()).collect();
        eprintln!(
            "jit: compiled with LLVM: {}",
            if names.is_empty() {
                "none".into()
            } else {
                names.join(", ")
            }
        );
        for (n, why) in &m.rejected {
            eprintln!("jit: interpreted: {n} ({why})");
        }
    }
    let r = it.run(&prog);
    if let Some(p) = it.harness.passes {
        eprintln!("budget: {} of {p} passes used", it.harness.used);
    } else if it.harness.used > 0 {
        eprintln!("budget: {} passes used (unlimited)", it.harness.used);
    }
    if it.native.is_some() {
        eprintln!("jit: {} native calls", it.native_calls);
    }
    let _ = Rc::new(());
    r.map(|_| ()).map_err(|d| d.render(path, src))
}

fn validate_run_args(args: &[String]) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--jit" | "--interpret" => i += 1,
            "--budget" | "--task" | "--allow" | "--log" => {
                if args.get(i + 1).is_none_or(|v| v.starts_with("--")) {
                    return Err(format!("{} needs a value", args[i]));
                }
                i += 2;
            }
            other => return Err(format!("unknown runtime argument `{other}`")),
        }
    }
    Ok(())
}

// ------------------------------------------------------------ edits2w

/// Convert the last ```edits block of a post-format answer into a Wisp program.
fn cmd_edits2w(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("edits2w needs a file")?;
    let sizes = flag(args, "--sizes").ok_or("edits2w needs --sizes, e.g. 600,32,32,6")?;
    let act = flag(args, "--act").unwrap_or_else(|| "tanh".into());
    let out_file = flag(args, "--out").unwrap_or_else(|| "edited.safetensors".into());
    let text = read(path)?;
    let block = text
        .rsplit("```edits")
        .next()
        .filter(|_| text.contains("```edits"))
        .ok_or("no ```edits block found")?;
    let block = block.split("```").next().unwrap_or("");
    // join continuation lines by bracket balance
    let mut stmts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for raw in block.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        for c in line.chars() {
            match c {
                '[' => depth += 1,
                ']' => depth -= 1,
                _ => {}
            }
        }
        cur.push_str(line);
        cur.push(' ');
        if depth <= 0 {
            stmts.push(cur.trim().to_string());
            cur.clear();
            depth = 0;
        }
    }
    let sizes_w = sizes.replace(',', " ");
    let mut out = format!(
        "// Converted from {path} by `wisp edits2w`.\n// The post's files store W as [in, out]; Wisp shows every weight as [out, in],\n// so the two indices of each W are swapped, and half-open slices a:b become a..b-1.\nnet = mlp([{sizes_w}], '{act}');\n\n"
    );
    let mut skipped = 0;
    for s in &stmts {
        match convert_stmt(s) {
            Ok(w) => {
                out += &w;
                out += "\n";
            }
            Err(e) => {
                skipped += 1;
                out += &format!("/* skipped: {s}  ({e}) */\n");
            }
        }
    }
    out += &format!("\nwsave(net, '{out_file}');\n");
    print!("{out}");
    eprintln!(
        "edits2w: {} statements converted, {skipped} skipped",
        stmts.len() - skipped
    );
    Ok(())
}

fn convert_stmt(s: &str) -> Result<String, String> {
    let (lhs, op, rhs) = ["+=", "-=", "*=", "="]
        .iter()
        .find_map(|op| {
            s.find(op)
                .map(|i| (s[..i].trim(), *op, s[i + op.len()..].trim()))
        })
        .ok_or("no assignment")?;
    let lb = lhs.find('[').ok_or("no index")?;
    let name = lhs[..lb].trim();
    let idx = lhs[lb + 1..lhs.rfind(']').ok_or("no ]")?].trim();
    let parts: Vec<&str> = idx.split(',').map(|p| p.trim()).collect();
    let conv: Vec<String> = parts
        .iter()
        .map(|p| slice_to_wisp(p))
        .collect::<Result<_, _>>()?;
    let is_matrix = name.starts_with('W') && conv.len() == 2;
    let (index, transpose_value) = if is_matrix {
        (format!("{}, {}", conv[1], conv[0]), true)
    } else {
        (conv.join(", "), false)
    };
    let value = value_to_wisp(rhs, transpose_value)?;
    let target = format!("net.{name}[{index}]");
    Ok(match op {
        "=" => format!("{target} = {value};"),
        "+=" => format!("{target} = {target} + {value};"),
        "-=" => format!("{target} = {target} - {value};"),
        _ => format!("{target} = {target} .* {value};"),
    })
}

fn slice_to_wisp(p: &str) -> Result<String, String> {
    if p == ":" {
        return Ok(":".into());
    }
    if !p.contains(':') {
        p.parse::<i64>().map_err(|_| format!("bad index `{p}`"))?;
        return Ok(p.to_string());
    }
    let f: Vec<&str> = p.split(':').collect();
    let a: i64 = if f[0].is_empty() {
        0
    } else {
        f[0].parse().map_err(|_| "bad start")?
    };
    let step: Option<i64> = if f.len() > 2 && !f[2].is_empty() {
        Some(f[2].parse().map_err(|_| "bad step")?)
    } else {
        None
    };
    let end = if f[1].is_empty() {
        "last".to_string()
    } else {
        let b: i64 = f[1].parse().map_err(|_| "bad stop")?;
        (b - 1).to_string()
    };
    Ok(match step {
        Some(s) if s != 1 => format!("{a}..{end} by {s}"),
        _ => format!("{a}..{end}"),
    })
}

fn value_to_wisp(v: &str, transpose: bool) -> Result<String, String> {
    let v = v.trim();
    if !v.starts_with('[') {
        v.parse::<f64>().map_err(|_| format!("bad value `{v}`"))?;
        return Ok(v.to_string());
    }
    let inner = v[1..v.len() - 1].trim();
    if inner.starts_with('[') {
        // 2-D list: rows of the post's first index
        let rows: Vec<String> = inner
            .split(']')
            .map(|r| {
                r.trim()
                    .trim_start_matches(',')
                    .trim()
                    .trim_start_matches('[')
                    .trim()
                    .to_string()
            })
            .filter(|r| !r.is_empty())
            .collect();
        let m = format!("[{}]", rows.join("; "));
        return Ok(if transpose { format!("{m}'") } else { m });
    }
    Ok(format!("[{inner}]"))
}

use std::io::{BufRead, Read, Write};
use wisp_python::{Session, MAX_SOURCE_BYTES};

fn bounded_read(reader: impl Read) -> Result<String, String> {
    let mut source = String::new();
    reader
        .take(MAX_SOURCE_BYTES as u64 + 1)
        .read_to_string(&mut source)
        .map_err(|e| e.to_string())?;
    if source.len() > MAX_SOURCE_BYTES {
        return Err("Python source limit exceeded".into());
    }
    Ok(source)
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    let script = match args {
        [mode, script] if mode == "python" => Some(script),
        [mode] if mode == "python-kernel" => None,
        _ => return Err("usage: wisp python <script.py>\n       wisp python-kernel".into()),
    };
    // Read only the host-selected script, with a bound before allocating its contents.
    let source = script
        .map(|path| {
            std::fs::File::open(path)
                .map_err(|e| format!("{path}: {e}"))
                .and_then(bounded_read)
        })
        .transpose()?;
    wisp_python::with_session(move |session| {
        if let Some(source) = source {
            session
                .execute(&source)
                .map(|result| print!("{}", result.output))
        } else {
            kernel(session)
        }
    })
}

fn kernel(session: &mut Session) -> Result<(), String> {
    // One JSON string containing Python source per line; no mounts, budgets,
    // snapshots or policy configuration are accepted over this channel.
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        let mut line = Vec::new();
        let n = (&mut input)
            .take((MAX_SOURCE_BYTES * 6 + 3) as u64)
            .read_until(b'\n', &mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(());
        }
        if line.len() > MAX_SOURCE_BYTES * 6 + 2 {
            return Err("Python kernel input limit exceeded".into());
        }
        let source: String = serde_json::from_slice(&line)
            .map_err(|e| format!("expected a JSON source string: {e}"))?;
        let (response, failed) = match session.execute(&source) {
            Ok(result) => (
                serde_json::json!({"ok": true, "output": result.output}),
                false,
            ),
            Err(error) => (serde_json::json!({"ok": false, "error": error}), true),
        };
        println!("{response}");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        if failed {
            return Err("Python kernel session terminated after failed execution".into());
        }
    }
}

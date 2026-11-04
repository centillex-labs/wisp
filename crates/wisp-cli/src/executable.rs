//! Linux embedded-runtime executables: a copy of wispc plus a versioned source trailer.
//! This is packaging, not whole-program native compilation. The kernel loads the
//! executable normally; our entry point reads its own trailer and runs the source
//! through the same interpreter and harness as `wisp`.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 16] = b"WISP-EXEC-v1\0\0\0\0";
const FOOTER: u64 = 24; // little-endian u64 payload length, then magic
const MAX_PAYLOAD: u64 = 64 * 1024 * 1024;
const USAGE: &str = "usage: wispc <file.wisp> -o <executable>\n\
Build a Linux executable with an embedded Wisp runtime and program source.\n\
Existing output files are refused.\n\
Executables run natively by default (requires Clang; no interpreter fallback).\n\
Use --interpret for runtime options: --budget N --log FILE.";

pub fn main_entry() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some((path, src)) = read_payload(&exe)? {
        if args == ["--help"] || args == ["-h"] {
            println!("Wisp embedded-runtime executable ({path})\nusage: program (native execution by default; requires Clang)\n   or: program --interpret [--budget N] [--log FILE]\nNo automatic interpreter fallback. Relative data/output paths use the current working directory.");
            return Ok(());
        }
        return crate::run_source(&path, &src, &args);
    }
    if args == ["--help"] || args == ["-h"] {
        println!("{USAGE}");
        return Ok(());
    }
    if args.len() != 3 || args[1] != "-o" || args[0].starts_with('-') {
        return Err(USAGE.into());
    }
    if !cfg!(target_os = "linux") {
        return Err("wispc executable packaging currently supports Linux only".into());
    }
    let src = fs::read_to_string(&args[0]).map_err(|e| format!("{}: {e}", args[0]))?;
    // Reject invalid source before creating any output. Do not execute it at build time.
    wisp_syntax::parse(&src).map_err(|d| d.render(&args[0], &src))?;
    let payload = serde_json::to_vec(&serde_json::json!({"path": args[0], "source": src}))
        .map_err(|e| e.to_string())?;
    if payload.len() as u64 > MAX_PAYLOAD {
        return Err("embedded source exceeds the 64 MiB payload limit".into());
    }
    package(&exe, Path::new(&args[2]), &payload)?;
    println!("built {} (embedded Wisp runtime)", args[2]);
    Ok(())
}

fn read_payload(exe: &Path) -> Result<Option<(String, String)>, String> {
    let mut f = File::open(exe).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    if len < FOOTER {
        return Ok(None);
    }
    f.seek(SeekFrom::End(-(FOOTER as i64)))
        .map_err(|e| e.to_string())?;
    let mut footer = [0; FOOTER as usize];
    f.read_exact(&mut footer).map_err(|e| e.to_string())?;
    if &footer[8..] != MAGIC {
        return Ok(None);
    }
    let size = u64::from_le_bytes(footer[..8].try_into().unwrap());
    if size > MAX_PAYLOAD || size > len - FOOTER {
        return Err("invalid Wisp executable payload length".into());
    }
    f.seek(SeekFrom::Start(len - FOOTER - size))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0; size as usize];
    f.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let data: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid Wisp executable payload: {e}"))?;
    let path = data["path"]
        .as_str()
        .ok_or("missing executable source path")?;
    let src = data["source"].as_str().ok_or("missing executable source")?;
    Ok(Some((path.into(), src.into())))
}

fn package(exe: &Path, out: &Path, payload: &[u8]) -> Result<(), String> {
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    // A private staging file in the destination directory. hard_link publishes
    // atomically without ever replacing an existing file (including the source).
    let mut staging = None;
    for n in 0..1000 {
        let path = parent.join(format!(".wispc-{}-{n}.tmp", std::process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => {
                staging = Some((path, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", out.display())),
        }
    }
    let (tmp, mut file) = staging.ok_or("cannot create executable staging file")?;
    let result = (|| -> std::io::Result<()> {
        std::io::copy(&mut File::open(exe)?, &mut file)?;
        file.write_all(payload)?;
        file.write_all(&(payload.len() as u64).to_le_bytes())?;
        file.write_all(MAGIC)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o755))?;
        }
        file.sync_all()?;
        drop(file);
        fs::hard_link(&tmp, out)?;
        Ok(())
    })();
    let _ = fs::remove_file(&tmp);
    result.map_err(|e| format!("{}: {e}", out.display()))
}

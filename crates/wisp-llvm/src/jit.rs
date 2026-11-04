//! Compile LLVM IR to a native shared library with clang and load it.
//!
//! This backend drives the LLVM 18 toolchain as a subprocess (`clang -O2 -x ir`),
//! which works wherever clang is installed, including systems without LLVM's C
//! headers. An in-process backend through `inkwell` can replace `compile_ir`
//! without changing callers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

pub struct NativeLib {
    lib: libloading::Library,
    pub dir: PathBuf,
    // Declared after the library, so unloading happens before cleanup.
    _temp: TempDir,
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Result<Self, String> {
        loop {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let dir = std::env::temp_dir().join(format!("wisp-jit-{}-{n}", std::process::id()));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&dir) {
                Ok(()) => return Ok(Self(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn clang() -> String {
    std::env::var("WISP_CLANG").unwrap_or_else(|_| "clang".into())
}

/// Write `ir` to a temporary directory and compile it with clang.
pub fn compile_ir(ir: &str, opt: &str) -> Result<NativeLib, String> {
    let temp = TempDir::new()?;
    let dir = temp.0.clone();
    let ll = dir.join("module.ll");
    let so = dir.join("module.so");
    std::fs::write(&ll, ir).map_err(|e| e.to_string())?;
    let out = Command::new(clang())
        .args([opt, "-shared", "-fPIC", "-Wno-override-module", "-x", "ir"])
        .arg(&ll)
        .arg("-o")
        .arg(&so)
        .arg("-lm")
        .output()
        .map_err(|e| format!("could not run clang ({e}); install clang 18 or set WISP_CLANG"))?;
    if !out.status.success() {
        return Err(format!(
            "clang failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let lib = unsafe { libloading::Library::new(&so) }.map_err(|e| e.to_string())?;
    Ok(NativeLib {
        lib,
        dir,
        _temp: temp,
    })
}

/// Compile IR to a relocatable object file for linking into other programs.
pub fn emit_object(ir: &str, out: &str) -> Result<(), String> {
    let temp = TempDir::new()?;
    let ll = temp.0.join("module.ll");
    std::fs::write(&ll, ir).map_err(|e| e.to_string())?;
    let o = Command::new(clang())
        .args(["-O2", "-c", "-fPIC", "-Wno-override-module", "-x", "ir"])
        .arg(&ll)
        .arg("-o")
        .arg(out)
        .output()
        .map_err(|e| e.to_string())?;
    if !o.status.success() {
        return Err(String::from_utf8_lossy(&o.stderr).into());
    }
    Ok(())
}

type ScalarFn = unsafe extern "C" fn(*const f64) -> f64;
type ForwardFn = unsafe extern "C" fn(*const f64, *mut f64);

/// Natively compiled scalar Wisp functions.
pub struct NativeScalar {
    _lib: Rc<NativeLib>,
    fns: HashMap<String, (ScalarFn, usize)>,
}

impl NativeScalar {
    pub fn load(lib: NativeLib, names: &[(String, usize)]) -> Result<Self, String> {
        let lib = Rc::new(lib);
        let mut fns = HashMap::new();
        for (n, arity) in names {
            let sym: libloading::Symbol<ScalarFn> =
                unsafe { lib.lib.get(format!("wisp_{n}\0").as_bytes()) }
                    .map_err(|e| e.to_string())?;
            fns.insert(n.clone(), (*sym, *arity));
        }
        Ok(NativeScalar { _lib: lib, fns })
    }
    pub fn has(&self, name: &str) -> bool {
        self.fns.contains_key(name)
    }
    pub fn arity(&self, name: &str) -> Option<usize> {
        self.fns.get(name).map(|f| f.1)
    }
    pub fn call(&self, name: &str, args: &[f64]) -> Option<f64> {
        let (f, arity) = self.fns.get(name)?;
        if args.len() != *arity {
            return None;
        }
        Some(unsafe { f(args.as_ptr()) })
    }
}

/// A whole scalar program. Its entry point executes native code only; the Rust
/// wrapper loads the module and translates a numeric failure record to a diagnostic.
pub struct NativeProgram {
    _lib: NativeLib,
    entry: unsafe extern "C" fn(*mut u64),
}

impl NativeProgram {
    pub fn load(lib: NativeLib) -> Result<Self, String> {
        let entry = unsafe {
            *lib.lib
                .get::<unsafe extern "C" fn(*mut u64)>(b"wisp_native_entry\0")
                .map_err(|e| e.to_string())?
        };
        Ok(Self { _lib: lib, entry })
    }

    pub fn run(&mut self) -> Result<(), wisp_syntax::Diag> {
        let mut result = [0u64; 3];
        unsafe {
            (self.entry)(result.as_mut_ptr());
        }
        let (code, message) = match result[0] {
            0 => return Ok(()),
            1 => ("E-ASSERT", "native assertion failed"),
            2 => ("E-UNDEFINED", "scalar variable or function output was read before assignment"),
            3 => ("E-SHAPE", "native range needs finite bounds, a nonzero finite step, and at most 2^53-1 iterations"),
            _ => ("E-JIT", "unknown native execution failure"),
        };
        Err(wisp_syntax::Diag::new(
            code,
            message,
            wisp_syntax::Span {
                start: result[1] as usize,
                end: result[2] as usize,
            },
        ))
    }
}

/// A natively compiled network forward pass.
pub struct NativeForward {
    _lib: NativeLib,
    f: ForwardFn,
    pub n_in: usize,
    pub n_out: usize,
}

impl NativeForward {
    pub fn load(lib: NativeLib, n_in: usize, n_out: usize) -> Result<Self, String> {
        let f: ForwardFn = unsafe {
            *lib.lib
                .get::<ForwardFn>(b"wisp_forward\0")
                .map_err(|e| e.to_string())?
        };
        Ok(NativeForward {
            _lib: lib,
            f,
            n_in,
            n_out,
        })
    }
    pub fn run(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n_in);
        let mut y = vec![0.0; self.n_out];
        unsafe { (self.f)(x.as_ptr(), y.as_mut_ptr()) };
        y
    }
}

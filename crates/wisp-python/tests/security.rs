//! These tests run the embedded interpreter directly: no worker, Python install or environment setup.
use wisp_python::{Session, MAX_SOURCE_BYTES};

// Match the CLI's explicit native-stack sizing, without environment overrides.
macro_rules! embedded_test {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() {
            wisp_python::with_session(|_| {
                $body;
                Ok(())
            })
            .unwrap();
        }
    };
}

embedded_test!(python_functions_operators_and_persistent_state, {
    let mut s = Session::new().unwrap();
    s.execute("a = tensor([1, 2, 3, 4], 2, 2)\ndef twice(x):\n    return x + x\nb = twice(a)\nc = a @ b\nassert c.values() == [14.0, 20.0, 30.0, 44.0]\nassert c.shape() == (2, 2)\nassert a.T.values() == [1., 3., 2., 4.]").unwrap();
    let result = s.execute("print(c.values())").unwrap();
    assert_eq!(result.output, "[14.0, 20.0, 30.0, 44.0]\n");
    s.close();
    assert!(s.execute("1").is_err());
});

embedded_test!(capabilities_are_denied_and_failures_close_session, {
    for code in [
        "open('/etc/passwd').read()",
        "open('/tmp/wisp-python-no-write', 'w')",
        "import os",
        "from pathlib import Path",
        "import subprocess",
        "import socket",
        "import ctypes",
        "import sys",
        "import _io",
        "import posix",
        "__import__('os')",
        "eval('1 + 1')",
        "exec('x = 1')",
        "compile('x = 1', '', 'exec')",
        "getattr(1, '__class__')",
        "globals()",
        "locals()",
        "vars()",
        "input()",
        "x = (1).__class__",
        "x = print.__self__",
        "x = tensor.__globals__",
        "'{0.__class__}'.format(1)",
        "wload('/etc/passwd')",
        "hf_download('anything')",
        "budget(999999)",
        "wisp_values(-1)",
        "wisp_zeros(2**63-1, 2**63-1)",
        "try:\n    while True: pass\nexcept:\n    pass",
    ] {
        let mut s = Session::new().unwrap();
        assert!(s.execute(code).is_err(), "unexpectedly permitted: {code}");
        assert!(s.execute("1 + 1").is_err(), "failed session was reused");
    }
});

embedded_test!(bytecode_and_host_resource_limits, {
    // Intentionally do NOT allocate unbounded Python memory: the embedded VM
    // has no hard heap/RSS limit. Such a test could OOM this entire test process.
    for code in [
        "while True:\n    pass",
        "while True: pass",
        "print('x' * 100_000)",
        "def f():\n    return f()\nf()",
        "a = zeros(1, 1)\nfor i in range(2000):\n    a.shape()",
        "a = zeros(256, 256)\nb = a @ a",
    ] {
        let mut s = Session::new().unwrap();
        let start = std::time::Instant::now();
        assert!(s.execute(code).is_err(), "unbounded execution: {code}");
        assert!(start.elapsed() < std::time::Duration::from_secs(10));
        assert!(s.execute("print('alive')").is_err());
    }
});

embedded_test!(host_limits_survive_multiple_feeds, {
    let mut s = Session::new().unwrap();
    for _ in 0..4 {
        s.execute("a = zeros(256, 256)").unwrap();
    }
    assert!(s.execute("zeros(1, 1)").is_err());
    let mut s = Session::new().unwrap();
    s.execute("print('x' * 40000)").unwrap();
    assert!(s.execute("print('x' * 40000)").is_err());
    let mut s = Session::new().unwrap();
    assert!(s.execute(&" ".repeat(MAX_SOURCE_BYTES + 1)).is_err());
    assert!(s.execute("1").is_err());
    let mut s = Session::new().unwrap();
    for _ in 0..64 {
        s.execute("x = 1").unwrap();
    }
    assert!(s.execute("x = 2").is_err());
});

embedded_test!(ordinary_python_and_print_keywords, {
    let mut s = Session::new().unwrap();
    let result = s.execute("values = [i * 2 for i in range(3)]\nfor i in range(3):\n    values[i] += 1\nassert values == [1, 3, 5]\nprint('ok', sum(values), sep=':', end='!')").unwrap();
    assert_eq!(result.output, "ok:9!");
    assert!(s.execute("print('x', file=1)").is_err());
});

embedded_test!(sessions_do_not_share_values_or_tensor_handles, {
    let mut first = Session::new().unwrap();
    first.execute("secret = 42\na = zeros(1, 1)").unwrap();
    let mut second = Session::new().unwrap();
    assert!(second.execute("wisp_values(0)").is_err());
    let mut second = Session::new().unwrap();
    assert!(second.execute("print(secret)").is_err());
    assert_eq!(first.execute("print(secret)").unwrap().output, "42\n");
});

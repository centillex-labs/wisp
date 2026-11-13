# Embedded Python in Wisp (experimental)

Wisp embeds **RustPython 0.6.0 directly in the Wisp process**. Python source is
compiled and executed by that interpreter; tensor operations use Wisp's existing
Rust `Tensor` implementation. The wrapper definitions are compiled into the Wisp
executable. There is **no Monty dependency, worker process, Python executable,
CPython library, Python home, or runtime installation step**.

This is the first frontend migration stage, not yet a full replacement for Wisp's
model-editing language. Existing `.wisp` programs remain supported.

## Build and run

Rust 1.95 or newer is required by the embedded interpreter (the resolved dependency
set may require a newer toolchain). Python support is enabled in normal CLI builds:

```sh
cargo build -p wisp-cli
./target/debug/wisp examples/restricted_python.py
# Equivalent explicit form:
./target/debug/wisp python examples/restricted_python.py
```

Copying the Wisp binary is sufficient for this Python frontend; standard platform
native-library requirements still apply. No Python files or libraries have to be
installed beside it. `--no-default-features` builds the legacy-only CLI if needed.

```python
a = tensor([1, 2, 3, 4], 2, 2)  # flat row-major numeric data
b = zeros(2, 2)
c = a + b                      # equal shapes only
product = a @ a                # matrix multiplication
transposed = a.T               # transpose property
assert a.shape() == (2, 2)
assert a.values() == [1.0, 2.0, 3.0, 4.0]
print(product.values())
```

Explicit `.add()`, `.matmul()` and `.transpose()` methods are also available.
These methods and operators call Rust synchronously in the same process.

## Persistent sessions

```sh
./target/debug/wisp python-kernel
```

Each input line is a JSON string containing Python source:

```json
"a = tensor([2], 1, 1)"
"print((a @ a).values())"
```

Responses contain `ok` and captured `output`, or `ok: false` and `error`.
Multiline Python is encoded using JSON escapes. Use `print` for output; final
expression values are not automatically rendered. Invalid protocol input or any
failed submission ends the session. This is distinct from the legacy kernel
protocol.

Rust callers use a synchronous, in-process API:

```rust,no_run
wisp_python::with_session(|session| {
    session.execute("a = tensor([2], 1, 1)")?;
    let execution = session.execute("print((a @ a).values())")?;
    assert_eq!(execution.output, "[4.0]\n");
    Ok(())
})?;
# Ok::<(), String>(())
```

`with_session` and the CLI use a **thread inside the Wisp process**, with a 16 MiB
native stack, and join it synchronously. RustPython's debug bootstrap exceeds the
usual 2 MiB Rust test-thread stack. This stack sizing does not create a subprocess,
RPC boundary, second executable, or crash isolation; Rust tensor callbacks remain
direct calls. Advanced embedders can use `Session::new()` on their own adequately
sized thread and keep the session there.

Python globals, tensors, host-call accounting and output accounting persist across
submissions. `close` (also called on drop) releases interpreter state and retained
tensors. No Tokio runtime or external runtime configuration is needed.

## Restricted language and capability policy

This frontend intentionally accepts a restricted surface of real Python. It uses
RustPython's parser and compiler, not a Wisp-written Python evaluator.

Supported constructs include functions, lambdas, assignments, conditionals,
`for`/`while`, lists, tuples, dictionaries, sets, comprehensions, indexing,
arithmetic, comparisons, assertions and f-strings.

The AST policy rejects imports, user-defined classes, private names,
exception handlers, context managers, decorators, annotations, generators,
async execution, dynamic scope declarations, and attributes outside the public
Wisp tensor API. Attribute assignment is also rejected. Exception handlers are
currently excluded so scripts cannot catch an execution-limit exception and
continue with tracing disabled. This is a restriction on accepted user source,
not a claim that RustPython itself lacks those language features.

The builtin dictionary is explicitly constructed from pure numeric/container
helpers and a captured `print`. It does not include `open`, `input`, `eval`,
`exec`, `compile`, `__import__`, `getattr`, `globals`, `locals` or `vars`.
Tensor wrappers share that same restricted builtin dictionary. Bootstrap-only
class/property helpers are removed before user code runs.

RustPython is compiled with `default-features = false` and only `compiler` and
`gc`. In particular, **`host_env`, `stdio`, `threading`, `importlib` and the general
standard library are not enabled**. RustPython's filesystem-backed `FileIO` and
host OS modules are behind `host_env`, not merely hidden from globals. The Ruff parser/AST dependency family is pinned together at 0.16.5 because later
0.16.x releases changed APIs used by this RustPython version. Build
feature unification matters: introducing another dependency that enables these
features changes this policy and requires a new security review. Check with:

```sh
cargo tree -p wisp-cli -e features -i rustpython-vm
```

The Rust host exposes only `wisp_tensor`, `wisp_zeros`, `wisp_transpose`,
`wisp_matmul`, `wisp_add`, `wisp_shape`, and `wisp_values`. These validate arity,
primitive input types, shapes, data lengths, handles, storage and computational
work before calling the tensor engine. Python convenience wrappers are not an
access-control boundary. Handles are session-local indices; a session is the
unit of data confidentiality.

No model files, downloads, network/process access, environment access or general
Wisp builtin dispatcher are exposed. `wload`, `wsave`, `run`, `gen`, `budget`,
`hf_download` and Wisp `getenv` are intentionally absent.

## Limits and their exact scope

| Resource | Limit |
|---|---:|
| Source per submission | 64 KiB |
| Accepted AST depth/nodes | 100 / 10,000 |
| Submissions per session | 64 |
| Captured output per session | 64 KiB |
| Host calls per session, including print | 1,000 |
| Trace events per submission | 100,000 |
| Cooperative elapsed execution check | 2 seconds |
| Python recursion depth | 100 |
| Elements per tensor | 65,536 |
| Retained tensor elements per session | 262,144 (2 MiB float64 payload) |
| Tensor handles per session | 256 |
| Numeric work per operation/session | 1,000,000 / 10,000,000 units |

All tensors are retained until session closure; deleting a wrapper does not
refund host accounting. Matrix work is conservatively charged as `m*k*n`.
Limits are private Rust state and cannot be changed with a Python assignment or
replenished by sending another submission. Any failure closes the session.

Native tracing enables opcode events and checks a host-owned counter/deadline.
This interrupts ordinary Python loops, including single-line loops, but **it is
not a hard wall-clock watchdog**. It does not preempt parsing, compilation, or a
single long-running native builtin operation. Bounded Wisp tensor callbacks have
separate work checks.

Output storage and diagnostics are capped, but conversion of a Python object to
its string representation happens before the output buffer check and may allocate
additional memory.

### Important security boundary

**This is a capability-restricted embedded interpreter, not a hard sandbox for
arbitrary hostile code.** It has no per-interpreter heap/RSS limit. Ordinary
Python operations such as multiplying a large string or constructing an enormous
container can exhaust Wisp's process memory. A native operation can exceed the
cooperative deadline, and an interpreter/native-code bug can crash the process.

The AST policy, reduced builtins and compile-time capability exclusions are
layered restrictions, not a proof against all interpreter escapes. The runtime
and dependency supply chain are trusted. This implementation has not undergone
an independent security audit. Do not execute adversarial code in a privileged
or secret-bearing process on the strength of these restrictions alone.

Strong availability and OS isolation require constraints around the entire Wisp
process (OS CPU/memory limits and deployment confinement), or further work on
RustPython's allocator and uncatchable execution accounting. That tradeoff is
explicit: Wisp remains a **single self-contained interpreter**, not a hidden
worker architecture. Legacy `.wisp` execution does not inherit these Python
capability restrictions.

## Verification

```sh
cargo test -p wisp-python
cargo test -p wisp-cli --test python
cargo test -p wisp-cli
```

All frontend tests now run normally, without ignored worker tests or environment
configuration. CLI tests run with an empty environment and working directory `/`,
exercising the binary without PATH/PYTHONHOME/runtime discovery. Tests cover tensor
operators, persistent state, denied capabilities, session isolation, bounded host
work/output, and cooperative loop termination. They intentionally do not trigger
unbounded Python allocations in the test process.

## Remaining migration work

* Model handles, authorized loading/saving and host-owned observation budgets.
* Tensor slicing/view/copy/mutation semantics, broadcasting and solvers.
* Large-model workload limits and cancellation; the initial tiny numerical quotas
  are not a model-inference solution.
* Broader Python syntax support only with tests preserving interruption and
  capability policy; no claim of CPython package compatibility.
* Allocator/execution hardening and adversarial review before making stronger
  sandbox guarantees or removing the legacy frontend.

References: [RustPython](https://github.com/RustPython/RustPython),
[rustpython-vm 0.6.0](https://docs.rs/rustpython-vm/0.6.0/rustpython_vm/).

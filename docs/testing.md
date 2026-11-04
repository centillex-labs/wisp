# Unit tests

Run the tests declared in a program:

```bash
wisp test examples/unit_tests.wisp
wisp test examples/unit_tests.wisp --budget 1
```

A test is a function whose name begins with `test_` (case-sensitive). It must
accept exactly one argument and declare no output values. The argument name is
conventional; `t` is not a keyword.

```wisp
function test_addition(t)
    assert(t.name == 'test_addition');
    assert(2 + 3 == 5, 'addition should return 5');
end
```

The context is a struct with metadata and bound methods:

- `t.name`: the current test function's name.
- `t.file`: the program path supplied to the command.

Use the existing `assert(condition [, message, ...])`, including inside helper
functions. A false or empty condition fails with `E-ASSERT`. A failure stops the
current test, not the suite. Other runtime errors also fail that test. Context
fields are metadata; changing them does not change the runner's test identity.
No special assertion method or additional `t` argument is required.

## Context methods

| Method | Behavior |
| --- | --- |
| `t.fprintf(format, ...)` | Write formatted test output; no automatic newline. |
| `t.error(format, ...)` | Record a failure with a source diagnostic, then continue. |
| `t.fail()` | Record a failure without a custom message, then continue. |
| `t.skip()` or `t.skip(format, ...)` | Stop the test immediately, unwinding helpers, and report a skip reason. |

Formatting follows ordinary `fprintf`. Pass the context or a bound method to a
helper when needed. Methods stay bound to their original test even when aliased.
Every `t.error` and `t.fail` is retained; later success or `t.skip` cannot erase a
failure. Assertions and runtime errors still stop execution immediately, retaining
any failures already recorded.

```wisp
function test_optional_model(t)
    if getenv('WISP_TEST_MODEL') == ''
        t.skip('set WISP_TEST_MODEL to enable this test');
    end
    t.fprintf('running %s from %s\n', t.name, t.file);
    assert(2 + 3 == 5);
end
```

A skipped test does not count as passed or failed. A suite containing only skipped
tests exits successfully, but its summary reports zero passes. If a test records a
failure before skipping, it counts as failed instead.

## Execution

- Discovery parses one supplied file; it does not scan directories or imports.
- All tests run in declaration order. Helpers are available to every test.
- Top-level statements are **not executed**. Put fixture construction, dataset
  declarations, budget limits, and protection rules inside tests or helpers.
- Each test gets fresh interpreter state, locals, and observation accounting.
  No network objects or registered datasets are shared across tests.
- `--budget N` is a per-test observation ceiling. `budget(N)` can tighten it, not
  raise it. Without a limit, observation usage is unlimited.
- File writes, caches, downloads, and other external side effects are **not**
  isolated or rolled back. Tests have the same access as normal programs.
- Duplicate function names and invalid test signatures fail discovery before any
  test executes. A file with no tests fails rather than reporting a passing suite.

The command prints captured program output, a `PASS`, `FAIL`, or `SKIP` line per test, and
a summary. Failures include the original source location and diagnostic on stderr.
Exit status is zero only when at least one test was found and none failed.

```text
PASS test_array_sum
PASS test_output_weights
2 passed, 0 failed, 2 total
```

`wisp program.wisp` continues to execute only top-level statements; it does not
automatically call tests. `--check` checks syntax but does not discover or execute
tests. This initial test command does not provide fixtures/hooks, filtering,
parallel execution, timeouts, or `--log` output.

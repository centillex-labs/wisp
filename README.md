# Wisp

Wisp is a language for humans and AI agents to perform **[Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)**. It lets you inspect and edit model weights directly, using assertions to test whether targeted changes improve model behavior.

## Bonsai examples

### Strawberry: load, edit, and generate

```bash
cargo build --release
./target/release/wisp strawberry.wisp
```

The self-contained [program](strawberry.wisp) caches Bonsai, runs inference, edits
four output rows, reloads the saved weights, and asserts that it generates `3`.
Native Rust CPU execution; no Python or task file. Allow about 8 GiB of available RAM.

Recorded one-token generation from the full vocabulary:

| Case | Before | After |
| --- | --- | --- |
| Count `r` in `strawberry` | `To` | **`3`** |
| Count `r` in `cherry` | `3` | `2` |
| Count `r` in `banana` | `3` | `0` |
| Count `r` in `orange` | `3` | `1` |
| Is 9 larger than 4? | `True` | `True` |
| Is 4 larger than 9? | `False` | `False` |

`To` was the original model's first token, not an incorrect numerical answer `2`.
The correct counts are supplied as training targets; all six inputs are used in
fitting, not held-out validation. This is a targeted repair, not a general counting
claim. The saved FP16 edit reproduced these outputs in a fresh process and passed
byte-exact patch reconstruction. [Exact prompts, results, and API](docs/strawberry.md).

### Repair boolean judgments

On Bonsai-1.7B’s FP16 export, we changed two output-token rows to repair character-count judgments using Wisp assertions.

| Cases | Before | After |
| --- | ---: | ---: |
| Repair | 6/12 | 12/12 |
| Held-out plural variants | 6/12 | 12/12 |
| Integer-comparison regression | 12/12 | 12/12 |

Core of the [program we ran](examples/bonsai_behavior_repair.wisp):

```wisp
base = wload('head_base.safetensors');
R = activations('repair');
G = activations('regression');
y = mod((0..rows(R)-1)', 2);

before = run(base, 'repair');
margin = before[:, 1] - before[:, 0];
H = [R; G];
residual = [(2*y-1)*0.3 - margin; zeros(rows(G), 1)];
change = H' * ((H*H' + 0.0001*eye(rows(H))) \ residual);

candidate = copy(base);
candidate.W1[1, :] += (change/2)';
candidate.W1[0, :] -= (change/2)';
assert(norm(candidate.W1-base.W1)/norm(base.W1) <= 0.5);

wsave(candidate, 'head_candidate.safetensors', 'budget', 1);
stored = wload('head_candidate.safetensors');
after = run(stored, 'repair');
[~, repaired] = max(after, [], 1);
assert(all(repaired == y));
assert(all((2*y-1).*(after[:,1]-after[:,0]) >= 0.25));
```

[Confirmation program](examples/bonsai_behavior_confirm.wisp):

```wisp
function scores = accuracy(model, items)
    Y = run(model, items);
    [~, predicted] = max(Y, [], 1);
    expected = mod((0..rows(Y)-1)', 2);
    scores = predicted == expected;
end

assert(confirm(base, stored, 'confirmation', @accuracy));
```

Constrained `False`/`True` classification, evaluated with an external CPU backend—not general generation improvement. The 4,092-entry edit breaks the two rows’ 1-bit structure; some confirmation margins are small. [Results and reproduction](docs/bonsai-behavior-repair.md).

### Preserve block scales while editing signs

The [sign-edit program](examples/bonsai_17.wisp) changed eight coefficients in layer 0, preserving every magnitude and block scale. Its helper `rebalance_blocks` is defined in the linked script.

```wisp
net = wload('model.safetensors', 'parts', 'model.layers.0.mlp.down_proj.weight');
D = net.L[0].mlp.down;
assert(rows(D) == 2048);
assert(cols(D) == 6144);

changed = rebalance_blocks(D, 128, 4);
assert(changed == 8);
```

A verified structural edit, not a demonstrated quality improvement. [Results](docs/bonsai-17.md).

## Original Weight-Space Programming research

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) — Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. Six experiments from the original post:

| Experiment | Original write-up, examples, and logs |
| --- | --- |
| Identify a game from its network weights | [Experiment 1](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game) |
| Identify what a diffusion model generates | [Experiment 2](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp2_diffusion) |
| Identify a language from middle transformer blocks | [Experiment 3](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp3_language) |
| Repair a failing car network | [Experiment 4](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car) |
| Write MinAtar networks from scratch | [Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch) |
| Write a 1.5-million-weight Go network | [Experiment 6](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch) |

Experiment 5 originals → Wisp ports:

- [Breakout](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/runs/breakout/reply/answer.md) → [`breakout.wisp`](examples/breakout.wisp)
- [Seaquest](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/runs/seaquest/reply/answer.md) → [`seaquest.wisp`](examples/seaquest.wisp)
- [Asterix](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/runs/asterix/reply/answer.md) → [`asterix.wisp`](examples/asterix.wisp)

Original example material is CC BY 4.0; Wisp source is MIT licensed.

## Build

```bash
cargo build --release
cargo test --release --workspace
./target/release/wisp examples/parity.wisp --check
./target/release/wisp examples/parity.wisp --budget 1
```

[Budget concepts](docs/budgets.md): observation passes and changed-weight limits.

## License

Wisp is [MIT licensed](LICENSE). The Bonsai checkpoint is Apache-2.0 licensed.

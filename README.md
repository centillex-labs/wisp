# Wisp

Wisp is a language for humans and AI agents to perform **[Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)**. It lets you inspect and edit model weights directly, using assertions to test whether targeted changes improve model behavior.

## Bonsai examples

### Strawberry: load, edit, and generate

```bash
cargo build --release
./target/release/wisp strawberry.wisp
```

[strawberry.wisp](strawberry.wisp) downloads and caches Bonsai, edits four output
rows, then reloads the saved weights and checks the generated answers. It runs on
the CPU in Rust and needs about 8 GiB of available RAM.

Each result is one generated token, selected from the full vocabulary:

| Case | Before | After |
| --- | --- | --- |
| Count `r` in `strawberry` | `To` | `3` |
| Count `r` in `cherry` | `3` | `2` |
| Count `r` in `banana` | `3` | `0` |
| Count `r` in `orange` | `3` | `1` |
| Is 9 larger than 4? | `True` | `True` |
| Is 4 larger than 9? | `False` | `False` |

The original strawberry response began with `To`. All six inputs are used to fit
the edit, with the correct counts supplied as targets. These results do not measure
accuracy on unseen questions. Reloading the saved FP16 weights reproduced the
outputs; patch reconstruction matched the saved file byte for byte.
[Prompts, results, and API](docs/strawberry.md).

### Repair boolean judgments

This earlier experiment changed two output-token rows in Bonsai-1.7B's FP16 export
to correct character-count judgments.

| Cases | Before | After |
| --- | ---: | ---: |
| Repair | 6/12 | 12/12 |
| Held-out plural variants | 6/12 | 12/12 |
| Integer-comparison regression | 12/12 | 12/12 |

[Edit program](examples/bonsai_behavior_repair.wisp):

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

Evaluation was restricted to choosing `False` or `True`, using an external CPU
backend. The 4,092-entry edit changes the rows' original 1-bit structure, and some
confirmation margins are small. [Recorded experiment](docs/bonsai-behavior-repair.md).

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

This test checks weight structure, not model behavior. [Results](docs/bonsai-17.md).

## Original Weight-Space Programming research

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab describes six experiments:

| Experiment | Original write-up, examples, and logs |
| --- | --- |
| Identify a game from its network weights | [Experiment 1](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game) |
| Identify what a diffusion model generates | [Experiment 2](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp2_diffusion) |
| Identify a language from middle transformer blocks | [Experiment 3](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp3_language) |
| Repair a failing car network | [Experiment 4](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car) |
| Write MinAtar networks from scratch | [Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch) |
| Write a 1.5-million-weight Go network | [Experiment 6](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch) |

Wisp ports of the Experiment 5 networks:

| Game | Original | Wisp |
| --- | --- | --- |
| Breakout | [Source](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/runs/breakout/reply/answer.md) | [breakout.wisp](examples/breakout.wisp) |
| Seaquest | [Source](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/runs/seaquest/reply/answer.md) | [seaquest.wisp](examples/seaquest.wisp) |
| Asterix | [Source](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/runs/asterix/reply/answer.md) | [asterix.wisp](examples/asterix.wisp) |

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

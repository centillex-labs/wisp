# Data from "Sparks of Weight-Space Programming"

The files in this folder come from the authors' released logs,
https://github.com/RikiyaT/lm_understand-logs (Experiment 5), by Rikiya Takehi,
Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL), licensed under
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

- `*_reference.npz`: the networks the model wrote for Breakout, Seaquest and Asterix
  (`exp5_write_from_scratch/networks/<game>/hand.npz`), unchanged.
- `*_design.md`: the model's answers containing the `edits` blocks
  (`exp5_write_from_scratch/runs/<game>/reply/answer.md`), unchanged.

The `../breakout.wisp`, `../seaquest.wisp`, and `../asterix.wisp` programs were generated from the answers by
`wisp edits2w`.

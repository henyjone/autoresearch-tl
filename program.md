# autoresearch-tl

This is an experiment to have the LLM autonomously research neural network architectures for underwater acoustic transmission loss (TL) prediction.

## Background

**Transmission Loss (TL)** measures how much acoustic energy is lost as sound propagates through the ocean. It depends on frequency, geometry (source/receiver positions and depths), sound speed profile (SSP), bathymetry, and seabed properties. Traditional physics models (Bellhop, RAM, KRAKEN) are accurate but slow. The goal is to train a fast neural network surrogate that predicts TL in milliseconds.

**Key physics intuition:**
- TL ≈ 20·log10(range) for spherical spreading — log-range is a critical feature
- Absorption increases with frequency² (Thorp formula) — log-frequency matters
- Sound channels (SSP minimum) can dramatically reduce TL at long ranges
- Shallow water has more bottom interactions → higher TL
- Soft bottoms (clay/silt) cause more loss than hard bottoms (rock/gravel)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch-tl/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-tl/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — this file, your instructions.
   - `prepare.py` — fixed constants, data generation, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-tl/data/` contains `train.npz`, `val.npz`, and `stats.npz`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, hidden dimensions, number of layers, activation functions, normalization, loss functions, learning rate schedules, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, feature encoding, and training constants (time budget, feature count, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_rmse` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_rmse (in dB).** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_rmse gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Architecture hints

The baseline is a simple MLP. Here are directions worth exploring:

- **Residual connections**: Skip connections between hidden layers
- **Feature engineering**: Log-transform frequency and range before the network sees them; compute derived features (wavelength, Thorp coefficient, critical angle) in the reserved feature slots
- **Separate encoders**: Process SSP (indices 8-27), bathymetry (28-47), and geometry (0-7) through separate encoder branches, then fuse
- **Attention over SSP**: Treat the 20 SSP depth samples as a sequence and apply self-attention
- **Loss function**: Try Huber loss, log-cosh loss, or MAE instead of MSE
- **Normalization**: BatchNorm, LayerNorm, or RMSNorm between layers
- **Wider/deeper**: The baseline is 4×256; try 6×512 or 8×128

## Output format

Once the script finishes it prints a summary like this:

```
---
val_rmse:         3.456789
training_seconds: 300.1
total_seconds:    305.2
peak_vram_mb:     1234.5
total_samples_M:  12.3
num_steps:        5000
num_params_K:     150.2
hidden_dim:       256
n_layers:         4
```

You can extract the key metric from the log file:

```
grep "^val_rmse:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_rmse	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_rmse achieved (e.g. 3.456789) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_rmse	memory_gb	status	description
a1b2c3d	3.456789	1.2	keep	baseline MLP 4x256
b2c3d4e	3.123456	1.3	keep	add residual connections
c3d4e5f	3.500000	1.2	discard	switch to ReLU activation
d4e5f6g	0.000000	0.0	crash	model too large (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-tl/mar23`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_rmse:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_rmse improved (lower), you "advance" the branch, keeping the git commit
9. If val_rmse is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the physics hints above, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

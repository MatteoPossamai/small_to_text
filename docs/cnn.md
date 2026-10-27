# CNN training and scoring

The selected model is the 32-neuron CNN with an eight-row window, trained for
500 epochs. Its input is 79 time rows × 20 frequency bands. Each of 32 sliding neurons sees an
8-row window; pooling averages groups of three positions, producing 768 values
for the `128 → 64 → 36` classifier.

Recreation status: verified. The exact recorded SHA-256 and
9,123 / 11,005 scoring result were reproduced.

The original winning file was lost when its worktree was removed. Its recorded
SHA-256 was
`882c37c59418972213ed45074b85e69bfc0f7a50f9b21b8bee81cbae1ac74b0c`.
The canonical recreated model path is
`network/4_cnn32_window8_20band_500.txt`; the status above will be updated only
after the recreated file and score are checked against the recorded result.

The older five-row checkpoints use `cnn5` (32 neurons) or `cnn16` (16 neurons).
The complete filename index and matching commands are in
[network/README.md](../network/README.md). `cnn8` is the selected eight-row build;
`cnn` keeps the same eight-row defaults for existing commands.

Build the executable:

```bash
cmake -S . -B build
cmake --build build --target cnn -j 2
```

Train a fresh model (use a new filename so an existing winner is not overwritten):

```bash
model=network/cnn32_window8_20band_500_retrained.txt
poppi_pop --cpu 1 -- ./build/cnn train weights/training "$model" \
  --epochs 500 --learning-rate 0.005 --seed 12345 --shift 4
./build/cnn score weights/training "$model"
./build/cnn score weights/scoring "$model"
```

Run long jobs inside tmux so they survive closing the terminal. Choose an
available CPU for `poppi_pop`.

The selected run used all 94,830 training recordings, with no validation split;
the scoring set has 11,005 recordings. Training saves checkpoints every ten
epochs and at completion. The learning rate is halved after ten epochs without
a 0.1% training-loss improvement, down to `0.00001`.

To resume, first copy the model to a new path, then pass that copy with
`--resume`:

```bash
cp --update=none-fail network/4_cnn32_window8_20band_500.txt network/cnn32_window8_resume600.txt
poppi_pop --cpu 1 -- ./build/cnn train weights/training network/cnn32_window8_resume600.txt \
  --resume --epochs 100 --learning-rate 0.00001
```

The command accepts `--epochs`, `--learning-rate`, `--seed`, `--shift`, and
`--resume`. It uses the configured 32-neuron, eight-row architecture. The
loader requires 20-band recordings and rejects non-finite values; do not pass
`weights_nonuniform40/`.

`inference_cnn` uses the same selected architecture and defaults to
`network/4_cnn32_window8_20band_500.txt`:

```bash
cmake --build build --target inference_cnn
./build/inference_cnn ~/Downloads/speech_commands_v0.02/one/00176480_nohash_0.wav
```

An optional second argument selects another compatible model file.

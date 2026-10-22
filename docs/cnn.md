# CNN training and scoring

The separate `cnn` executable uses existing encoded recordings from `weights/`.
It does not change the ordinary `train` and `scoring` commands.

With the current configuration, the model is:

```text
79 time rows × 20 bands
    → 16 sliding neurons, each seeing five rows
    → average groups of three positions
    → 400 → 128 → 64 → 36 word predictions
```

The architecture is selected in `src/cnn_main.cpp`. Epochs, learning rate, seed,
and maximum time shift are command-line options. `./build/cnn --help` lists them.

Build:

```bash
cmake -S . -B build
cmake --build build --target cnn -j 2
```

Train and then grade on both datasets (choose an available CPU):

```bash
poppi_pop --cpu 1 -- bash -ec '
  model=network/cnn_20band_500.txt
  ./build/cnn train weights/training "$model" \
    --epochs 500 --learning-rate 0.005 --seed 12345 --shift 4
  ./build/cnn score weights/training "$model"
  ./build/cnn score weights/scoring "$model"
'
```

Run the wrapper inside tmux if it should survive closing the terminal.

Training initializes a new model and refuses to overwrite an existing model.
It saves a checkpoint every ten epochs and at completion. It halves the learning
rate after ten epochs without a 0.1% improvement in mean training loss, down to
0.00001 (or the initial rate if lower). Scoring reports overall accuracy and
per-label results, with the weakest labels first.

Continue an existing model for 100 additional epochs:

```bash
./build/cnn train weights/training network/cnn_20band_500.txt \
  --resume --epochs 100 --learning-rate 0.001
```

Resuming restores learned weights and biases. The learning-rate schedule and
random sequence start again; this is not an exact continuation of an interrupted
run. Use `--shift 0` to disable random time shifting.

The loader checks exact sample sizes and rejects non-finite values. This build
expects 20-band recordings, so do not pass `weights_nonuniform40/`. CNN model
files use the `CNN_V1` format and are scored with this executable.

The program uses every sample in the supplied training directory. It does not
create a validation split or choose the best validation checkpoint automatically.

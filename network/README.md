## Index of Networks and how to run them

## Build

To build the repo you can use:

```shell
cmake -S . -B build
cmake --build build
```

This will build all into `./build` folder.

## Compile audios

We use [Google Speech Command Dataset](https://huggingface.co/datasets/google/speech_commands). In order to be able to run the scoring, you will need to compile them to the input of training. You can do it with:

```shell
./build/compile_audios /path/to/dataset ./weights
```

This will create what is required.

## Networks

### 1. Fully connected network - 77.23%

```bash
./build/scoring network/1_mlp_timeshift_20band_300.txt weights/scoring
```

### 2. CNN, 16 neurons, 5 rows - 81.59%

```bash
./build/cnn16 score weights/scoring network/2_cnn16_window5_20band_500.txt
```

### 3. CNN, 32 neurons, 5 rows, 500 epochs - 82.43%

```bash
./build/cnn5 score weights/scoring network/3_cnn32_window5_20band_500.txt
```

### 3. CNN, 32 neurons, 5 rows, 600 epochs - 82.45%

```bash
./build/cnn5 score weights/scoring network/3_cnn32_window5_20band_600.txt
```

### 4. CNN, 32 neurons, 8 rows, 500 epochs - 82.90% (recreation pending)

```bash
./build/cnn8 score weights/scoring network/4_cnn32_window8_20band_500.txt
```

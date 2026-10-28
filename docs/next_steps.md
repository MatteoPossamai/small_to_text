# What comes after 82.90%?

**My recommendation: finish the reproduction and release, give 85% one bounded
attempt, then build a three-second recognizer for known words.** That is a useful
learning step toward transcription without changing everything at once.

## 1. Give 85% a fair attempt

The [recorded results](evolution.md) show:

| What we know | What it suggests |
|---|---|
| 96.26% training, 82.90% scoring | Learning unfamiliar recordings deserves attention. |
| Eight-row model: 82.90% at 500 epochs, 82.82% at 600 | More training has not solved this. |
| Earlier log/mel features performed worse | Keep the current features for this comparison. |

85% needs **232 additional correct recordings** on the same 11,005-recording
scoring set. It is a worthwhile experiment, not a guaranteed result.

First, separate the data used for learning, choosing experiments, and final scoring:

```text
Training:    84,843 word recordings → learn weights
Validation:   9,981 word recordings → choose changes and stopping point
Scoring:     11,005 word recordings → report the final result
```

These word counts come from the original dataset lists. Currently, all validation
recordings are inside training, alongside six background files. Restore
`validation_list.txt`, keep speakers separate, and **train fresh models**: existing
checkpoints already saw those validation recordings. This three-way split follows
the [official training guide](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#validation).

Run one reference and two separate changes, using the same split and seed:

| Run | Change | Intuition |
|---|---|---|
| Reference | Current eight-row CNN | Measure the new baseline. |
| A | Occasionally hide a small time strip or frequency band during training | Learn despite missing clues. |
| B | Add a small penalty for large weights, called L2 | Discourage overly specific solutions. |

A adapts [SpecAugment's masking idea](https://arxiv.org/abs/1904.08779) to our
features; B uses [L2 regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization).
Neither has been tested here. Save the best validation checkpoint. Compare with
the fresh reference, since it uses fewer training recordings than the historical
82.90% model. Repeat a promising winner with another seed, then score the chosen
model. The scoring set has already guided earlier choices; fresh recordings would
provide an additional independent check.

Before training, inspect the scorer's existing confusion summary and listen to
representative validation mistakes. If neither change helps validation, move on.

## 2. Make three seconds useful

The current [WAV reader](../src/wav_file.hpp) stops after one second, and
[inference](../src/inference_cnn.cpp) returns one label.
Increasing the input length alone still gives one label.

Start with **two or three known words, spoken with pauses**:

```text
3-second WAV: "one … two … three"
        ↓
Overlapping one-second windows
        ↓
Existing CNN → stable predictions → separate word events
        ↓
"one two three"
```

Reject silence and uncertain predictions; prevent one word from firing repeatedly.
Test `"one … one"` explicitly so duplicate suppression preserves real repetitions.
This follows the [official streaming-command approach](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#streaming-accuracy).

Join isolated clips first to check the mechanics, keeping source speakers in
their original splits. Then test real three-second recordings, varying pauses,
word positions, and noise. Include background-only audio: the current scoring
set contains **zero background examples**.

Count wrong, missing, and extra words, plus fully correct sequences. Joined clips
do not establish performance on naturally connected speech.

## 3. Later, learn sequences directly

For speech without deliberate pauses, a next learning project is a network that
produces predictions throughout time, trained with **CTC**: a training rule that
uses the transcript without needing each word's timestamp.

```text
Time predictions: one one _ two _ one
Decoded words:    one two one

Time predictions: one _ one
Decoded words:    one one
```

Adjacent repeats merge before blanks disappear. `_` means “emit nothing now”; it
does not necessarily mean silence. This requires sequence training and a changed
output structure. See the [original CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

Begin with digits. Arbitrary sentences later need outputs for letters or word chunks
and sentence recordings paired with transcripts, such as
[LibriSpeech](https://www.openslr.org/12). That is a larger project than recognizing
three familiar words.

# Evolution

## CNN comparisons

The selected configuration is the 32-neuron CNN with an eight-row time
window, trained from scratch for 500 epochs using seed `12345`. It reached
96.26% on training data and 82.90% on the 11,005-recording scoring set
(9,123 correct). The file's recorded SHA-256 is
`882c37c59418972213ed45074b85e69bfc0f7a50f9b21b8bee81cbae1ac74b0c`.
The original file was lost with its worktree. A seeded recreation matched
the exact original checksum and scoring result. The selected artifact is
`network/4_cnn32_window8_20band_500.txt`.

| Configuration | Training accuracy | Scoring accuracy | Notes |
|---|---:|---:|---|
| 16 neurons, 5-row window, 500 epochs | 91.72% | 81.59% | Earlier baseline |
| 32 neurons, 5-row window, 500 epochs | - | 82.43% | Training result not recorded |
| 32 neurons, 5-row window, 600 epochs | 96.12% | 82.45% | 500-epoch run continued 100 epochs |
| 32 neurons, 5-row window, 700 epochs | 96.14% | 82.45% | Continued 200 epochs; seed `12346` |
| 32 neurons, 8-row window, 250 epochs | 94.31% | 82.76% | Fresh run |
| **32 neurons, 8-row window, 500 epochs** | **96.26%** | **82.90%** | **Selected; 9,123 / 11,005** |
| 32 neurons, 8-row window, 600 epochs | 96.28% | 82.82% | Continued 100 epochs; scoring fell slightly |
| 32 neurons, 12-row window, 300 epochs | 94.67% | 82.20% | Fresh run |
| 32 neurons, 12-row window, +200 epochs | - | - | User stopped after 59/200 additional epochs; no completed 500-epoch result |

These are exploratory single-seed comparisons, not a broad benchmark. The
12-row extension was stopped after its completed 300-epoch score trailed the
selected eight-row model. The later masking and L2 experiment below reached the 85% scoring goal.

## Audio features

Three 300-epoch runs used the same 1,580 → 128 → 64 → 36 network, seed `12345`,
sorted recording order, ordinary sampling, and random time shifts up to 50 ms.

| Features | Dataset | Training accuracy | Scoring accuracy |
|---|---|---:|---:|
| Equal-width bands, square-root energy | `weights/` | 84.63% | **77.08%** |
| Equal-width bands, `log10(1 + energy)` | `weights_log/` | 67.02% | 64.22% |
| Overlapping mel bands, `log10(1 + energy)` | `weights_mel/` | 77.63% | 72.85% |

Original features performed best in this comparison. The current encoder uses
square-root band energy with per-clip min-max scaling; it is not the provisional
logarithm version. Keep the original features as the baseline. This comparison
does not rule out other log or mel configurations.

## Masking and L2 comparison

These fresh runs used seed `12345`, 84,849 training files and the same
9,981-recording validation set. Scoring used the original 11,005 recordings.

| Training | Epochs | Validation | Scoring |
|---|---:|---:|---:|
| Baseline | 250 | 83.35% | 81.99% |
| L2 (0.0001) | 250 | 84.72% | 83.24% |
| Masking | 250 | 85.11% | 83.93% |
| **Masking + L2 (0.0001)** | **300** | **86.18%** | **85.06%** |

The combined model correctly classified 9,361 / 11,005 recordings. It is
`network/5_cnn32_window8_mask_l2_300.txt`. These are single-seed results.

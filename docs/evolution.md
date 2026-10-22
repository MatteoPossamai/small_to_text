# Evolution

## Audio feature comparison

Three 300-epoch runs: same 1580 → 128 → 64 → 36 network, seed 12345,
sorted recording order, ordinary sampling, and random time shifts up to 50 ms.
Learning-rate settings and the training/scoring split were unchanged.

| Features | Dataset | Training accuracy | Scoring accuracy |
|---|---|---:|---:|
| Original equal-width bands, square-root energy | `weights/` | 84.63% | **77.08%** |
| Equal-width bands, `log10(1 + energy)` | `weights_log/` | 67.02% | 64.22% |
| Overlapping mel bands, `log10(1 + energy)` | `weights_mel/` | 77.63% | 72.85% |

All variants used 20 bands. Original features performed best in this comparison;
neither log-only nor mel + log improved accuracy. Keep original features as the
baseline. This does not rule out other log/mel configurations.

An earlier original-feature run scored 77.23%, with different random starting
conditions. The goal remains 85% scoring accuracy; it has not been reached.

The working encoder still contains the provisional logarithm change. The
original-feature result above used the existing `weights/` files.

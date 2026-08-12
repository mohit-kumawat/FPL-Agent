# Ablation ladder (autosub-aware scoring, policy-gated transfers)

Seed 20260812; 5000 bootstrap resamples; CIs are 90% over per-GW paired differences. Season totals are noisy — the CI is the evidence.

## 2024-25

- `ppg`: **2101** pts (65 moves, 112 hit pts)
- `+fixtures`: **2213** pts (59 moves, 88 hit pts)
- `+minutes`: **2204** pts (65 moves, 112 hit pts)
- `full`: **2288** pts (48 moves, 52 hit pts)
- `full-greedy`: **2332** pts (52 moves, 60 hit pts)

| rung | Δtotal | per-GW Δ | 90% CI | verdict |
|---|---|---|---|---|
| ppg → +fixtures | +112 | +2.95 | [-0.45, +6.39] | consistent with noise |
| +fixtures → +minutes | -9 | -0.24 | [-3.42, +3.05] | consistent with noise |
| +minutes → full | +84 | +2.21 | [-1.32, +5.79] | consistent with noise |
| full → full-greedy | +44 | +1.16 | [-1.34, +3.63] | consistent with noise |
| ppg → full | +187 | +4.92 | [+1.03, +8.71] | earns its keep |

## 2023-24

- `ppg`: **2128** pts (68 moves, 124 hit pts)
- `+fixtures`: **2147** pts (63 moves, 108 hit pts)
- `+minutes`: **2203** pts (65 moves, 116 hit pts)
- `full`: **2366** pts (54 moves, 68 hit pts)
- `full-greedy`: **2320** pts (58 moves, 84 hit pts)

| rung | Δtotal | per-GW Δ | 90% CI | verdict |
|---|---|---|---|---|
| ppg → +fixtures | +19 | +0.50 | [-2.76, +3.71] | consistent with noise |
| +fixtures → +minutes | +56 | +1.47 | [-1.45, +4.39] | consistent with noise |
| +minutes → full | +163 | +4.29 | [+1.16, +7.42] | earns its keep |
| full → full-greedy | -46 | -1.21 | [-3.11, +0.68] | consistent with noise |
| ppg → full | +238 | +6.26 | [+2.05, +10.34] | earns its keep |


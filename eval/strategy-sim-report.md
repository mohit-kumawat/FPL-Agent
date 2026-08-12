# Strategy-return simulation (point-in-time, no hindsight)

## 2024-25
- **agent strategy**: 2232 pts (48 transfers, 52 hit pts, 60.1 raw pts/GW)
- **with real matchday rules (autosubs + vice)**: 2288 pts (+56 from 18 autosubs, vice used 0x)
- hold-GW1-squad baseline: 1884 pts (autosub-aware 1901)
- transfer engine added: +348 pts (autosub-aware +387)
- note: no chips in either arm; real managers gain ~30-60 pts/season from chips on top

## 2023-24
- **agent strategy**: 2267 pts (54 transfers, 68 hit pts, 61.4 raw pts/GW)
- **with real matchday rules (autosubs + vice)**: 2366 pts (+99 from 25 autosubs, vice used 1x)
- hold-GW1-squad baseline: 1939 pts (autosub-aware 1972)
- transfer engine added: +328 pts (autosub-aware +394)
- note: no chips in either arm; real managers gain ~30-60 pts/season from chips on top

## Lambda-grid robustness (Model A recency vs Model B ridge)

### 2024-25 (test GW27-38)
top-11 actual points by weight (higher=better):
top11_ens_0.1      5.258
top11_A_recency    5.258
top11_ens_0.2      4.977
top11_ens_0.3      4.682
top11_ens_0.4      4.614
top11_ens_0.5      4.258
top11_ens_0.6      4.189
top11_ens_0.7      4.098
top11_ens_0.8      4.023
top11_ens_0.9      3.962
top11_B_ridge      3.705

### 2025-26 (test GW27-38)
top-11 actual points by weight (higher=better):
top11_B_ridge      4.424
top11_ens_0.9      4.235
top11_ens_0.6      4.144
top11_ens_0.8      3.955
top11_ens_0.7      3.947
top11_ens_0.5      3.902
top11_ens_0.3      3.841
top11_ens_0.4      3.833
top11_ens_0.2      3.720
top11_ens_0.1      3.606
top11_A_recency    3.523

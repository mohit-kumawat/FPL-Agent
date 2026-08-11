# Strategy-return simulation (point-in-time, no hindsight)

## 2024-25
- **agent strategy**: 2284 pts (43 transfers, 32 hit pts, 60.9 raw pts/GW)
- hold-GW1-squad baseline: 1885 pts
- transfer engine added: +399 pts
- reference: winner ~2,810 | top-10k ~2,450 | overall average ~2,100

## 2023-24
- **agent strategy**: 2195 pts (53 transfers, 72 hit pts, 59.7 raw pts/GW)
- hold-GW1-squad baseline: 1944 pts
- transfer engine added: +251 pts
- reference: winner ~2,799 | overall average ~2,030
- note (both seasons): no chips, no autosubs in either arm; chips typically add 30-60 pts
- observation: 72 hit points in 2023/24 vs 32 in 2024/25 — the hit threshold may deserve a higher bar; logged as hypothesis

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

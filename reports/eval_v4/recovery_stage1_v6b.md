# Recovery Stage 1: calibration experiments on saved scores

Selection on validation only (out-of-fold for fitted calibrators). Test columns are shown for understanding; only the PRIMARY row is a gate.

## Sanity: identity rows reproduce the reported numbers


## Gates

- **current lora_v6b alone (sanity)**: lora_v4:ident / max / chunk @ +2.5  vs  baseline:ident / max / chunk @ +7.0: test contract F1 0.771 -> 0.852, diff +0.081 (95% CI +0.024 to +0.139) -> **PASS**
- **PRIMARY: selected candidate vs approved reference**: lora_v4:shift / max / chunk @ -2.0  vs  baseline:ident / max / chunk @ +7.0: test contract F1 0.771 -> 0.867, diff +0.096 (95% CI +0.039 to +0.151) -> **PASS**
- **STRESS: selected candidate vs best baseline variant**: lora_v4:shift / max / chunk @ -2.0  vs  baseline:platt / max / contract @ -1.0: test contract F1 0.799 -> 0.867, diff +0.068 (95% CI +0.016 to +0.126) -> **PASS**

## Secondary metrics of the selected candidate (reported, not gates)

- Chunk macro P/R/F1: 0.790 / 0.814 / 0.794; difference vs approved reference +0.194 (95% CI +0.150 to +0.237)
- Contract macro P/R/F1: 0.867 / 0.875 / 0.867
- Falsely flagged chunks per contract: 0.88; wrong clause flags per contract (of 6): 0.22
- Excluding Liquidated Damages: chunk F1 0.829, contract F1 0.916
- Average precision (ranking, threshold-free) vs raw zero-shot margins: chunk 0.711 -> 0.851 (diff +0.140, CI +0.095 to +0.186); contract 0.847 -> 0.918 (diff +0.071, CI +0.029 to +0.110)

| Clause | chunk F1 | contract F1 | blend weight v4 | blend weight baseline | intercept |
|---|---|---|---|---|---|
| Anti-Assignment | 0.847 | 0.991 | +0.86 | +0.19 | -1.90 |
| Auto-Renewal | 0.852 | 0.873 | +0.09 | +0.46 | -5.29 |
| Governing Law | 0.987 | 1.000 | +1.06 | +0.00 | -0.20 |
| Limitation of Liability | 0.854 | 0.966 | +0.68 | +0.13 | -2.21 |
| Liquidated Damages | 0.583 | 0.571 | +0.50 | +0.24 | -4.64 |
| Non-Compete | 0.667 | 0.783 | +0.73 | +0.12 | -3.51 |
| Termination for Convenience | 0.769 | 0.885 | +1.29 | +0.08 | -1.68 |

## Is the selection stable? (validation only, 5 other fold assignments)

| fold seed | selected | best val contract F1: v4:ident / v4:shift / v4:platt / blend:platt |
|---|---|---|
| 1 | lora_v4:shift | 0.866 / 0.888 / 0.869 / 0.870 |
| 2 | lora_v4:shift | 0.866 / 0.888 / 0.856 / 0.865 |
| 3 | lora_v4:shift | 0.866 / 0.888 / 0.858 / 0.869 |
| 4 | lora_v4:shift | 0.866 / 0.888 / 0.861 / 0.865 |
| 5 | lora_v4:shift | 0.866 / 0.888 / 0.860 / 0.867 |

## All variants (sorted by validation contract-F1)

| candidate | rule | threshold objective | t | val chunk F1 | val contract F1 | test chunk F1 | test contract F1 |
|---|---|---|---|---|---|---|---|
| lora_v4:shift | max | chunk | -2.0 | 0.804 | 0.888 | 0.794 | 0.867 |
| lora_v4:shift | max | contract | -2.0 | 0.804 | 0.888 | 0.794 | 0.867 |
| lora_v4:ident | max | contract | +2.0 | 0.768 | 0.866 | 0.773 | 0.858 |
| lora_v4:ident | max | chunk | +2.5 | 0.791 | 0.866 | 0.763 | 0.852 |
| blend:platt | max | contract | -1.0 | 0.789 | 0.863 | 0.793 | 0.845 |
| lora_v4:platt | max | chunk | -1.0 | 0.780 | 0.861 | 0.795 | 0.865 |
| lora_v4:platt | max | contract | -1.0 | 0.780 | 0.861 | 0.795 | 0.865 |
| blend:platt | max | chunk | -0.5 | 0.795 | 0.854 | 0.797 | 0.861 |
| blend:platt | max | zero | +0.0 | 0.789 | 0.828 | 0.806 | 0.870 |
| lora_v4:ident | max | zero | +0.0 | 0.721 | 0.824 | 0.751 | 0.835 |
| blend:platt | top2 | contract | -3.0 | 0.675 | 0.813 | 0.711 | 0.843 |
| lora_v4:platt | top2 | contract | -3.0 | 0.664 | 0.787 | 0.713 | 0.834 |
| lora_v4:ident | top2 | contract | -2.0 | 0.617 | 0.780 | 0.667 | 0.783 |
| baseline:platt | max | contract | -1.0 | 0.633 | 0.764 | 0.684 | 0.799 |
| lora_v4:platt | max | zero | +0.0 | 0.705 | 0.756 | 0.714 | 0.788 |
| baseline:ident | max | contract | +6.5 | 0.556 | 0.737 | 0.586 | 0.764 |
| baseline:platt | max | zero | +0.0 | 0.654 | 0.734 | 0.677 | 0.778 |
| baseline:platt | max | chunk | +0.0 | 0.654 | 0.734 | 0.677 | 0.778 |
| baseline:platt | top2 | contract | -3.0 | 0.531 | 0.731 | 0.588 | 0.766 |
| baseline:ident | max | chunk | +7.0 | 0.571 | 0.729 | 0.600 | 0.771 |
| baseline:ident | top2 | contract | +3.5 | 0.458 | 0.722 | 0.510 | 0.759 |
| baseline:ident | top2 | zero | +0.0 | 0.251 | 0.668 | 0.290 | 0.695 |
| lora_v4:shift | top2 | contract | -3.0 | 0.744 | 0.668 | 0.761 | 0.610 |
| lora_v4:ident | top2 | zero | +0.0 | 0.721 | 0.658 | 0.751 | 0.600 |
| baseline:ident | top2 | chunk | +7.0 | 0.571 | 0.650 | 0.600 | 0.681 |
| baseline:ident | max | zero | +0.0 | 0.251 | 0.649 | 0.290 | 0.682 |
| lora_v4:platt | top2 | chunk | -1.0 | 0.780 | 0.560 | 0.795 | 0.593 |
| lora_v4:shift | max | zero | +0.0 | 0.502 | 0.539 | 0.447 | 0.511 |
| lora_v4:shift | top2 | chunk | -2.0 | 0.804 | 0.529 | 0.794 | 0.520 |
| blend:platt | top2 | chunk | -0.5 | 0.795 | 0.528 | 0.797 | 0.551 |
| blend:platt | top2 | zero | +0.0 | 0.789 | 0.484 | 0.806 | 0.483 |
| lora_v4:platt | top2 | zero | +0.0 | 0.705 | 0.441 | 0.714 | 0.418 |
| baseline:platt | top2 | zero | +0.0 | 0.654 | 0.437 | 0.677 | 0.443 |
| baseline:platt | top2 | chunk | +0.0 | 0.654 | 0.437 | 0.677 | 0.443 |
| lora_v4:ident | top2 | chunk | +2.5 | 0.791 | 0.382 | 0.763 | 0.405 |
| lora_v4:shift | top2 | zero | +0.0 | 0.502 | 0.255 | 0.447 | 0.228 |

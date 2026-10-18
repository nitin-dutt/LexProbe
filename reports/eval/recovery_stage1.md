# Recovery Stage 1: calibration experiments on saved scores

Selection on validation only (out-of-fold for fitted calibrators). Test columns are shown for understanding; only the PRIMARY row is a gate.

## Sanity: identity rows reproduce the reported numbers

- baseline: threshold +7.0 (expected +7.0), test contract F1 0.739 (expected 0.739) -> OK
- lora_v4: threshold +4.5 (expected +4.5), test contract F1 0.720 (expected 0.720) -> OK

## Gates

- **current v4 (sanity)**: lora_v4:ident / max / chunk @ +4.5  vs  baseline:ident / max / chunk @ +7.0: test contract F1 0.739 -> 0.720, diff -0.019 (95% CI -0.094 to +0.054) -> **FAIL**
- **PRIMARY: selected candidate vs approved reference**: blend:platt / max / chunk @ -0.5  vs  baseline:ident / max / chunk @ +7.0: test contract F1 0.739 -> 0.828, diff +0.089 (95% CI +0.018 to +0.156) -> **PASS**
- **STRESS: selected candidate vs best baseline variant**: blend:platt / max / chunk @ -0.5  vs  baseline:platt / max / contract @ -1.0: test contract F1 0.773 -> 0.828, diff +0.055 (95% CI +0.006 to +0.107) -> **PASS**

## Secondary metrics of the selected candidate (reported, not gates)

- Chunk macro P/R/F1: 0.807 / 0.770 / 0.781; difference vs approved reference +0.205 (95% CI +0.146 to +0.249)
- Contract macro P/R/F1: 0.847 / 0.820 / 0.828
- Falsely flagged chunks per contract: 0.62; wrong clause flags per contract (of 6): 0.21
- Excluding Liquidated Damages: chunk F1 0.817, contract F1 0.894
- Average precision (ranking, threshold-free) vs raw zero-shot margins: chunk 0.693 -> 0.813 (diff +0.120, CI +0.075 to +0.156); contract 0.821 -> 0.861 (diff +0.039, CI -0.005 to +0.083)

| Clause | chunk F1 | contract F1 | blend weight v4 | blend weight baseline | intercept |
|---|---|---|---|---|---|
| Auto-Renewal | 0.833 | 0.873 | +0.07 | +0.46 | -5.42 |
| Governing Law | 0.987 | 1.000 | +0.56 | +0.12 | -3.01 |
| Limitation of Liability | 0.846 | 0.966 | +0.60 | +0.04 | -2.11 |
| Liquidated Damages | 0.600 | 0.500 | +0.26 | +0.30 | -5.81 |
| Non-Compete | 0.638 | 0.762 | +0.40 | +0.18 | -3.41 |
| Termination for Convenience | 0.783 | 0.868 | +0.99 | +0.08 | -2.50 |

## Is the selection stable? (validation only, 5 other fold assignments)

| fold seed | selected | best val contract F1: v4:ident / v4:shift / v4:platt / blend:platt |
|---|---|---|
| 1 | blend:platt | 0.739 / 0.758 / 0.777 / 0.819 |
| 2 | blend:platt | 0.739 / 0.758 / 0.780 / 0.828 |
| 3 | blend:platt | 0.739 / 0.758 / 0.730 / 0.816 |
| 4 | blend:platt | 0.739 / 0.758 / 0.766 / 0.833 |
| 5 | blend:platt | 0.739 / 0.758 / 0.762 / 0.827 |

## All variants (sorted by validation contract-F1)

| candidate | rule | threshold objective | t | val chunk F1 | val contract F1 | test chunk F1 | test contract F1 |
|---|---|---|---|---|---|---|---|
| blend:platt | max | chunk | -0.5 | 0.761 | 0.820 | 0.781 | 0.828 |
| blend:platt | max | contract | -0.5 | 0.761 | 0.820 | 0.781 | 0.828 |
| blend:platt | max | zero | +0.0 | 0.756 | 0.795 | 0.785 | 0.842 |
| lora_v4:platt | max | chunk | -1.5 | 0.675 | 0.784 | 0.707 | 0.785 |
| lora_v4:platt | max | contract | -1.5 | 0.675 | 0.784 | 0.707 | 0.785 |
| blend:platt | top2 | contract | -3.0 | 0.621 | 0.768 | 0.660 | 0.773 |
| lora_v4:shift | max | contract | -0.5 | 0.661 | 0.758 | 0.696 | 0.751 |
| lora_v4:ident | max | chunk | +4.5 | 0.655 | 0.739 | 0.672 | 0.720 |
| lora_v4:ident | max | contract | +4.5 | 0.655 | 0.739 | 0.672 | 0.720 |
| lora_v4:shift | max | chunk | +0.5 | 0.677 | 0.736 | 0.678 | 0.712 |
| baseline:platt | max | contract | -1.0 | 0.603 | 0.733 | 0.673 | 0.773 |
| lora_v4:shift | max | zero | +0.0 | 0.663 | 0.732 | 0.703 | 0.750 |
| lora_v4:platt | top2 | contract | -2.5 | 0.586 | 0.720 | 0.655 | 0.733 |
| lora_v4:platt | max | zero | +0.0 | 0.668 | 0.707 | 0.666 | 0.730 |
| baseline:platt | max | zero | +0.0 | 0.627 | 0.705 | 0.664 | 0.756 |
| baseline:platt | max | chunk | +0.0 | 0.627 | 0.705 | 0.664 | 0.756 |
| baseline:platt | top2 | contract | -3.0 | 0.506 | 0.699 | 0.568 | 0.733 |
| baseline:ident | max | contract | +6.5 | 0.517 | 0.698 | 0.561 | 0.731 |
| lora_v4:ident | top2 | contract | -0.5 | 0.534 | 0.697 | 0.590 | 0.720 |
| lora_v4:ident | max | zero | +0.0 | 0.554 | 0.693 | 0.607 | 0.733 |
| baseline:ident | max | chunk | +7.0 | 0.531 | 0.693 | 0.576 | 0.739 |
| baseline:ident | top2 | contract | +3.5 | 0.424 | 0.684 | 0.483 | 0.725 |
| lora_v4:ident | top2 | zero | +0.0 | 0.554 | 0.674 | 0.607 | 0.690 |
| baseline:ident | top2 | zero | +0.0 | 0.242 | 0.626 | 0.281 | 0.655 |
| baseline:ident | top2 | chunk | +7.0 | 0.531 | 0.617 | 0.576 | 0.649 |
| lora_v4:shift | top2 | contract | -3.0 | 0.598 | 0.612 | 0.666 | 0.623 |
| baseline:ident | max | zero | +0.0 | 0.242 | 0.604 | 0.281 | 0.645 |
| lora_v4:platt | top2 | chunk | -1.5 | 0.675 | 0.514 | 0.707 | 0.546 |
| blend:platt | top2 | chunk | -0.5 | 0.761 | 0.466 | 0.781 | 0.481 |
| blend:platt | top2 | zero | +0.0 | 0.756 | 0.438 | 0.785 | 0.420 |
| lora_v4:shift | top2 | zero | +0.0 | 0.663 | 0.434 | 0.703 | 0.428 |
| lora_v4:ident | top2 | chunk | +4.5 | 0.655 | 0.402 | 0.672 | 0.393 |
| baseline:platt | top2 | zero | +0.0 | 0.627 | 0.390 | 0.664 | 0.402 |
| baseline:platt | top2 | chunk | +0.0 | 0.627 | 0.390 | 0.664 | 0.402 |
| lora_v4:platt | top2 | zero | +0.0 | 0.668 | 0.388 | 0.666 | 0.374 |
| lora_v4:shift | top2 | chunk | +0.5 | 0.677 | 0.378 | 0.678 | 0.380 |

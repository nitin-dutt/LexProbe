# Full-document evaluation — baseline vs lora_v4

Dataset v3 **test** split: **76 contracts** (75 near-duplicate groups), **2,995 chunks**, every chunk scored for all 6 clauses: 268 positive and 17,689 negative (chunk, clause) pairs (**1.5% prevalence**; 13 partial pairs excluded). A clause is flagged when its first-token margin exceeds the threshold.

## Summary: operating points

| System @ threshold | Chunk P | Chunk R | Chunk F1 | Falsely flagged chunks / contract | Contract P | Contract R | Contract F1 | Wrong clause flags / contract |
|---|---|---|---|---|---|---|---|---|
| baseline @ +0.0 | 0.173 | 0.931 | **0.281** | 19.1 | 0.521 | 0.971 | **0.645** | 2.05 |
| lora_v4 @ +0.0 | 0.509 | 0.934 | **0.607** | 4.8 | 0.645 | 0.978 | **0.733** | 1.37 |
| baseline @ +7.0 | 0.539 | 0.735 | **0.576** | 3.3 | 0.743 | 0.814 | **0.739** | 0.53 |
| lora_v4 @ +4.5 | 0.767 | 0.659 | **0.672** | 0.5 | 0.810 | 0.717 | **0.720** | 0.26 |

## Natural threshold 0 (baseline +0.0, lora_v4 +0.0)

### Chunk level, every chunk of every contract

| Clause | Positives | baseline P | baseline R | baseline F1 | lora_v4 P | lora_v4 R | lora_v4 F1 |
|---|---|---|---|---|---|---|---|
| Auto-Renewal | 34 | 0.330 | 0.971 | 0.493 | 0.395 | 1.000 | 0.567 |
| Governing Law | 74 | 0.259 | 1.000 | 0.411 | 0.974 | 1.000 | 0.987 |
| Limitation of Liability | 82 | 0.121 | 1.000 | 0.215 | 0.740 | 0.939 | 0.828 |
| Liquidated Damages ⚠️ | 12 | 0.047 | 0.917 | 0.090 | 0.053 | 1.000 | 0.101 |
| Non-Compete | 33 | 0.163 | 0.909 | 0.276 | 0.418 | 0.848 | 0.560 |
| Termination for Convenience | 33 | 0.116 | 0.788 | 0.202 | 0.474 | 0.818 | 0.600 |
| **Macro** | | 0.173 | 0.931 | **0.281** | 0.509 | 0.934 | **0.607** |

- Macro-F1 baseline **0.281** (95% CI 0.245–0.319) → lora_v4 **0.607** (95% CI 0.560–0.648); difference +0.326 (95% CI +0.294 to +0.356), one-sided p = 0.0000
- Negative pairs flagged: baseline 1,451 (8.20%) vs lora_v4 363 (2.05%)

### Contract level: is each clause flagged in each contract?

| Clause | Positives | baseline P | baseline R | baseline F1 | lora_v4 P | lora_v4 R | lora_v4 F1 |
|---|---|---|---|---|---|---|---|
| Auto-Renewal | 31 | 0.612 | 0.968 | 0.750 | 0.517 | 1.000 | 0.681 |
| Governing Law | 62 | 0.925 | 1.000 | 0.961 | 1.000 | 1.000 | 1.000 |
| Limitation of Liability | 43 | 0.662 | 1.000 | 0.796 | 0.935 | 1.000 | 0.966 |
| Liquidated Damages ⚠️ | 7 | 0.120 | 0.857 | 0.211 | 0.125 | 1.000 | 0.222 |
| Non-Compete ⚠️ | 20 | 0.400 | 1.000 | 0.571 | 0.655 | 0.950 | 0.776 |
| Termination for Convenience ⚠️ | 25 | 0.410 | 1.000 | 0.581 | 0.639 | 0.920 | 0.754 |
| **Macro** | | 0.521 | 0.971 | **0.645** | 0.645 | 0.978 | **0.733** |

- Macro-F1 baseline **0.645** (0.586–0.697) → lora_v4 **0.733** (0.683–0.774); difference +0.088 (95% CI +0.057 to +0.122)
- Contracts with no wrong clause flag: baseline 8% vs lora_v4 14%
- Highest-scoring chunk (the one the API quotes) really is the clause: baseline 91.9% vs lora_v4 96.8% of correctly flagged contract-clauses

## Thresholds tuned on validation (baseline +7.0, lora_v4 +4.5)

### Chunk level, every chunk of every contract

| Clause | Positives | baseline P | baseline R | baseline F1 | lora_v4 P | lora_v4 R | lora_v4 F1 |
|---|---|---|---|---|---|---|---|
| Auto-Renewal | 34 | 0.903 | 0.824 | 0.862 | 0.682 | 0.882 | 0.769 |
| Governing Law | 74 | 0.771 | 1.000 | 0.871 | 0.986 | 0.986 | 0.986 |
| Limitation of Liability | 82 | 0.325 | 0.915 | 0.479 | 0.944 | 0.829 | 0.883 |
| Liquidated Damages ⚠️ | 12 | 0.179 | 0.583 | 0.275 | 0.429 | 0.500 | 0.462 |
| Non-Compete | 33 | 0.433 | 0.788 | 0.559 | 0.704 | 0.576 | 0.633 |
| Termination for Convenience | 33 | 0.625 | 0.303 | 0.408 | 0.857 | 0.182 | 0.300 |
| **Macro** | | 0.539 | 0.735 | **0.576** | 0.767 | 0.659 | **0.672** |

- Macro-F1 baseline **0.576** (95% CI 0.516–0.620) → lora_v4 **0.672** (95% CI 0.590–0.723); difference +0.097 (95% CI +0.046 to +0.141), one-sided p = 0.0000
- Negative pairs flagged: baseline 253 (1.43%) vs lora_v4 36 (0.20%)

### Contract level: is each clause flagged in each contract?

| Clause | Positives | baseline P | baseline R | baseline F1 | lora_v4 P | lora_v4 R | lora_v4 F1 |
|---|---|---|---|---|---|---|---|
| Auto-Renewal | 31 | 0.964 | 0.871 | 0.915 | 0.763 | 0.935 | 0.841 |
| Governing Law | 62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Limitation of Liability | 43 | 0.843 | 1.000 | 0.915 | 1.000 | 1.000 | 1.000 |
| Liquidated Damages ⚠️ | 7 | 0.238 | 0.714 | 0.357 | 0.273 | 0.429 | 0.333 |
| Non-Compete ⚠️ | 20 | 0.581 | 0.900 | 0.706 | 0.824 | 0.700 | 0.757 |
| Termination for Convenience ⚠️ | 25 | 0.833 | 0.400 | 0.541 | 1.000 | 0.240 | 0.387 |
| **Macro** | | 0.743 | 0.814 | **0.739** | 0.810 | 0.717 | **0.720** |

- Macro-F1 baseline **0.739** (0.669–0.799) → lora_v4 **0.720** (0.647–0.782); difference -0.019 (95% CI -0.094 to +0.054)
- Contracts with no wrong clause flag: baseline 55% vs lora_v4 75%
- Highest-scoring chunk (the one the API quotes) really is the clause: baseline 95.8% vs lora_v4 96.8% of correctly flagged contract-clauses

## Threshold-free ranking quality: average precision (AP)

| Clause | Chunk AP baseline | Chunk AP lora_v4 | Contract AP baseline | Contract AP lora_v4 |
|---|---|---|---|---|
| Auto-Renewal | 0.915 | 0.882 | 0.968 | 0.926 |
| Governing Law | 0.932 | 0.981 | 1.000 | 1.000 |
| Limitation of Liability | 0.756 | 0.947 | 0.991 | 1.000 |
| Liquidated Damages | 0.584 | 0.577 | 0.563 | 0.519 |
| Non-Compete | 0.592 | 0.593 | 0.756 | 0.735 |
| Termination for Convenience | 0.379 | 0.719 | 0.650 | 0.871 |
| **Macro** | **0.693** | **0.783** | **0.821** | **0.842** |

- Chunk-level macro AP difference +0.090 (95% CI +0.034 to +0.143); contract-level +0.021 (95% CI -0.044 to +0.079)

## Validation sweep — baseline (selected threshold +7.0)

| Threshold | Chunk P | Chunk R | Chunk F1 | Contract F1 |
|---|---|---|---|---|
| -2.0 | 0.062 | 0.968 | 0.116 | 0.558 |
| -1.0 | 0.100 | 0.937 | 0.177 | 0.580 |
| +0.0 | 0.145 | 0.904 | 0.242 | 0.604 |
| +1.0 | 0.194 | 0.876 | 0.307 | 0.630 |
| +2.0 | 0.235 | 0.844 | 0.352 | 0.632 |
| +3.0 | 0.291 | 0.825 | 0.408 | 0.646 |
| +4.0 | 0.334 | 0.802 | 0.444 | 0.666 |
| +5.0 | 0.394 | 0.787 | 0.482 | 0.677 |
| +6.0 | 0.457 | 0.742 | 0.502 | 0.685 |
| +7.0 | 0.514 | 0.736 | 0.531 | 0.693 |

## Validation sweep — lora_v4 (selected threshold +4.5)

| Threshold | Chunk P | Chunk R | Chunk F1 | Contract F1 |
|---|---|---|---|---|
| -2.0 | 0.380 | 0.921 | 0.481 | 0.666 |
| -1.0 | 0.420 | 0.902 | 0.520 | 0.683 |
| +0.0 | 0.462 | 0.883 | 0.554 | 0.693 |
| +1.0 | 0.527 | 0.847 | 0.597 | 0.717 |
| +2.0 | 0.572 | 0.838 | 0.627 | 0.720 |
| +3.0 | 0.621 | 0.774 | 0.638 | 0.735 |
| +4.0 | 0.707 | 0.687 | 0.636 | 0.730 |
| +4.5 | 0.773 | 0.648 | 0.655 | 0.739 |
| +5.0 | 0.814 | 0.599 | 0.640 | 0.704 |
| +6.0 | 0.758 | 0.436 | 0.512 | 0.550 |


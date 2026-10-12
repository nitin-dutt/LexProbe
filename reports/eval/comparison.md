# Test-set evaluation — baseline vs lora_v4

Dataset v3 test split: **804 queries** (268 positive / 536 negative), 68 contract groups. Decision = first-token logit `Found` vs `None`.

| Clause | Pos/Neg | baseline P | baseline R | baseline F1 | lora_v4 P | lora_v4 R | lora_v4 F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| Auto-Renewal | 34/68 | 1.000 | 0.971 | 0.985 | 0.971 | 1.000 | 0.986 | +0.000 |
| Governing Law | 74/148 | 0.851 | 1.000 | 0.919 | 1.000 | 1.000 | 1.000 | +0.081 |
| Limitation of Liability | 82/164 | 0.695 | 1.000 | 0.820 | 1.000 | 0.939 | 0.969 | +0.149 |
| Liquidated Damages ⚠️ | 12/24 | 0.733 | 0.917 | 0.815 | 1.000 | 1.000 | 1.000 | +0.185 |
| Non-Compete | 33/66 | 0.882 | 0.909 | 0.896 | 1.000 | 0.848 | 0.918 | +0.023 |
| Termination for Convenience | 33/66 | 0.839 | 0.788 | 0.812 | 0.964 | 0.818 | 0.885 | +0.073 |
| **Macro** | | 0.833 | 0.931 | **0.875** | 0.989 | 0.934 | **0.960** | **+0.085** |

⚠️ = fewer than 30 positive test queries — per-class numbers are unreliable.

## Significance (paired, resampling contract groups)

- baseline macro-F1 **0.875** (95% CI 0.822–0.904)
- lora_v4 macro-F1 **0.960** (95% CI 0.936–0.980)
- Difference **+0.085** (95% CI +0.051 to +0.149), one-sided p = 0.0000 (2000 draws)
- McNemar: only lora_v4 correct on 68 queries, only baseline correct on 12; two-sided p = 1.20e-10

## Format compliance (unconstrained top token is `Found`/`None`)

- baseline: 100.0% · lora_v4: 100.0%

## Evidence quality (quotes generated for predicted positives)

| System | Quotes | Grounded in source | Exact substring | TP span-F1 vs gold |
|---|---|---|---|---|
| baseline | 318 | 72.3% | 71.4% | 0.790 |
| lora_v4 | 254 | 100.0% | 100.0% | 0.911 |

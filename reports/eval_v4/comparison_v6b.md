# Test-set evaluation — baseline vs lora_v6b

Dataset v4 test split: **1077 queries** (359 positive / 718 negative), 73 contract groups. Decision = first-token logit `Found` vs `None`.

| Clause | Pos/Neg | baseline P | baseline R | baseline F1 | lora_v6b P | lora_v6b R | lora_v6b F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| Anti-Assignment | 91/182 | 0.853 | 0.956 | 0.902 | 1.000 | 0.835 | 0.910 | +0.009 |
| Auto-Renewal | 34/68 | 1.000 | 0.971 | 0.985 | 1.000 | 0.882 | 0.938 | -0.048 |
| Governing Law | 74/148 | 0.822 | 1.000 | 0.902 | 1.000 | 1.000 | 1.000 | +0.098 |
| Limitation of Liability | 82/164 | 0.719 | 1.000 | 0.837 | 0.986 | 0.890 | 0.936 | +0.099 |
| Liquidated Damages ⚠️ | 12/24 | 0.611 | 0.917 | 0.733 | 0.923 | 1.000 | 0.960 | +0.227 |
| Non-Compete | 33/66 | 0.938 | 0.909 | 0.923 | 1.000 | 0.939 | 0.969 | +0.046 |
| Termination for Convenience | 33/66 | 1.000 | 0.788 | 0.881 | 1.000 | 0.848 | 0.918 | +0.037 |
| **Macro** | | 0.849 | 0.934 | **0.881** | 0.987 | 0.914 | **0.947** | **+0.067** |

⚠️ = fewer than 30 positive test queries — per-class numbers are unreliable.

## Significance (paired, resampling contract groups)

- baseline macro-F1 **0.881** (95% CI 0.828–0.909)
- lora_v6b macro-F1 **0.947** (95% CI 0.921–0.968)
- Difference **+0.067** (95% CI +0.032 to +0.120), one-sided p = 0.0000 (2000 draws)
- McNemar: only lora_v6b correct on 76 queries, only baseline correct on 25; two-sided p = 3.72e-07

## Format compliance (unconstrained top token is `Found`/`None`)

- baseline: 100.0% · lora_v6b: 100.0%

## Evidence quality (quotes generated for predicted positives)

| System | Quotes | Grounded in source | Exact substring | TP span-F1 vs gold |
|---|---|---|---|---|
| baseline | 415 | 74.5% | 73.3% | 0.776 |
| lora_v6b | 326 | 100.0% | 99.7% | 0.901 |

# Model comparison on the test split (76 contracts, 2995 chunks)

Raw-margin average precision (threshold-free), clauses: Auto-Renewal, Governing Law, Limitation of Liability, Liquidated Damages, Non-Compete, Termination for Convenience. 95% ranges from 400 contract resamples. Reference: `v5`.

| model | macro AP | 95% range | diff vs reference | 95% range of diff | verdict |
|---|---|---|---|---|---|
| v5 | 0.829 | 0.769 to 0.884 | +0.000 | +0.000 to +0.000 | - |
| v6 | 0.813 | 0.747 to 0.875 | -0.016 | -0.047 to +0.002 | same within noise |
| v6b | 0.837 | 0.770 to 0.890 | +0.008 | -0.017 to +0.033 | same within noise |
| baseline | 0.692 | 0.607 to 0.765 | -0.137 | -0.208 to -0.083 | worse |

Per clause (point estimates):

| clause | v5 | v6 | v6b | baseline |
|---|---|---|---|---|
| Auto-Renewal (positives 34) | 0.953 | 0.887 | 0.956 | 0.915 |
| Governing Law (positives 74) | 0.976 | 0.974 | 0.947 | 0.930 |
| Limitation of Liability (positives 82) | 0.947 | 0.946 | 0.937 | 0.756 |
| Liquidated Damages (positives 12) | 0.708 | 0.680 | 0.709 | 0.584 |
| Non-Compete (positives 33) | 0.611 | 0.606 | 0.652 | 0.591 |
| Termination for Convenience (positives 33) | 0.779 | 0.786 | 0.820 | 0.378 |

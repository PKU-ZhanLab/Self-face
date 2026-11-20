# Self-face memory analysis

The code computes participant-level correlations for Figure 5C and model-data
discrepancies for Figure 6 of the self-face memory study.

| File | Analysis | Input |
| --- | --- | --- |
| [`analysis/inference.py`](analysis/inference.py) | Pearson correlations and Holm correction (Figure 5C) | A participant-level table with `cfmt` and the summary measures. Tests within a family share the same complete cases and use the specified test directions. |
| [`analysis/model_readouts.py`](analysis/model_readouts.py) | Normalized global (G), visual (V), and semantic (S) discrepancies (Figure 6) | Precomputed category codes, masks, and weights, or predicted and reference summaries: four category proportions, six visual RFI values, and two semantic RFI values. |

## Run the example

[`example.py`](example.py) generates synthetic inputs for both analyses and
checks the resulting statistics and discrepancy scores. Tested with Python 3.10.

```sh
python -m pip install -r requirements.txt
python example.py
```

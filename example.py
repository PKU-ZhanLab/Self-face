"""Demonstrate both analysis modules with synthetic inputs only."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.inference import pearson_family
from analysis.model_readouts import (
    figure5_visual_rfi,
    semantic_status_rfi,
    three_block_errors,
    weighted_category_proportions,
)


def correlation_example():
    table = pd.DataFrame({
        "cfmt": [0, 1, 2, 3, 4, 5],
        "visual_summary": [0, 2, 1, 4, 3, 5],
        "semantic_summary": [5, 3, 4, 1, 2, np.nan],
        "reliance_summary": [1, 0, 2, 4, 3, 5],
    })
    results = pearson_family(
        table, ["visual_summary", "semantic_summary", "reliance_summary"],
        "synthetic_example",
        alternatives={"visual_summary": "greater", "semantic_summary": "less"},
    )
    assert (results["n"] == 5).all()
    assert (results["p_holm"] >= results["p_raw"]).all()
    return results


def discrepancy_example():
    vertices = np.arange(12)
    evidence = SimpleNamespace(
        mask=vertices < 11,
        raw_weights=np.ones(12),
        figure5_pairs=[(vertices == i, vertices == i + 1) for i in range(0, 12, 2)],
        figure5_denominators=np.full(6, 2),
        semantic_full_masks=[vertices < 6, vertices >= 6],
        semantic_reported=[False, True],
    )
    reference = np.tile(np.arange(4), 3)
    candidates = np.stack([reference, reference])
    candidates[1, 0] = 1
    ref_u, ref_r = semantic_status_rfi(evidence, reference)
    cand_u, cand_r = semantic_status_rfi(evidence, candidates)
    results = three_block_errors(
        weighted_category_proportions(evidence, candidates),
        figure5_visual_rfi(evidence, candidates),
        cand_u, cand_r,
        target_global_proportions=weighted_category_proportions(evidence, reference)[0],
        target_visual_rfi=figure5_visual_rfi(evidence, reference)[0],
        target_semantic_unreported_rfi=ref_u[0],
        target_semantic_reported_rfi=ref_r[0],
    )
    np.testing.assert_allclose(results, [[0, 0, 0], [1/11, 0.5/np.sqrt(6), 1/(6*np.sqrt(2))]])
    return pd.DataFrame(results, columns=["G", "V", "S"], index=["Reference", "Modified"])


if __name__ == "__main__":
    print("Synthetic correlation example")
    print(correlation_example().to_string(index=False))
    print("\nSynthetic discrepancy example")
    print(discrepancy_example().to_string())

"""Participant-level inference with explicitly scoped Holm families."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def pearson_family(frame, columns, family, alternatives=None):
    """Use one cohort and declared directions; retain two-sided P for audit."""
    alternatives = alternatives or {}
    complete = frame.dropna(subset=['cfmt', *columns])
    if not np.isfinite(complete[['cfmt', *columns]]).all().all():
        raise ValueError('Non-finite correlation input')
    rows = []
    for column in columns:
        alternative = alternatives.get(column, 'two-sided')
        result = pearsonr(complete.cfmt, complete[column], alternative=alternative)
        two_sided = pearsonr(complete.cfmt, complete[column])
        rows.append(dict(metric=column, n=len(complete), r=float(result.statistic),
                         p_two_sided=float(two_sided.pvalue), p_raw=float(result.pvalue),
                         alternative=alternative, family=family))
    table = pd.DataFrame(rows)
    table['p_holm'] = multipletests(table.p_raw, method='holm')[1]
    table['family_size'] = len(columns)
    return table


def holm_label(value):
    return 'P(Holm)' + (' < 0.001' if value < .001 else f' = {value:.3f}')

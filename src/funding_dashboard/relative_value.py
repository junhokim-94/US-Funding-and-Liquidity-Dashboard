from __future__ import annotations

import numpy as np
import pandas as pd

from funding_dashboard.features import robust_zscore


RV_SERIES = {
    "SOFR vs IORB": "SOFR_IORB_BP",
    "TGCR vs IORB": "TGCR_IORB_BP",
    "EFFR vs IORB": "EFFR_IORB_BP",
    "1M AA financial CP vs Treasury": "AA_FIN_CP_1M_TSY_BP",
    "1M AA nonfinancial CP vs Treasury": "AA_NONFIN_CP_1M_TSY_BP",
    "1M A2/P2 vs AA CP": "A2P2_AA_1M_BP",
    "3M AA financial CP vs Treasury": "AA_FIN_CP_3M_TSY_BP",
    "3M AA nonfinancial CP vs Treasury": "AA_NONFIN_CP_3M_TSY_BP",
    "3M A2/P2 vs AA CP": "A2P2_AA_3M_BP",
}


def build_rv_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in RV_SERIES.items():
        if col not in features or features[col].dropna().empty:
            continue
        s = features[col].dropna()
        latest = s.iloc[-1]
        row = {"instrument_or_spread": label, "current_bp": latest}
        for name, days in [("1w", 5), ("1m", 21), ("3m", 63)]:
            row[f"change_{name}_bp"] = latest - s.iloc[-days - 1] if len(s) > days else np.nan
            z = robust_zscore(s, window=max(126, days * 4), min_periods=min(60, max(20, days)))
            row[f"z_{name}"] = z.iloc[-1] if not z.dropna().empty else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["relative_value_signal"] = table[["z_1w", "z_1m", "z_3m"]].mean(axis=1)
    table["interpretation"] = np.select(
        [table["relative_value_signal"] >= 1.5, table["relative_value_signal"] <= -1.5],
        ["Wide / elevated compensation", "Tight / low compensation"],
        default="Near normal",
    )
    return table.sort_values("relative_value_signal", ascending=False)

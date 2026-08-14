from __future__ import annotations

import numpy as np
import pandas as pd


COMPONENT_SPECS = {
    "unsecured_funding": {
        "absolute": ["EFFR_ADMIN_BP", "OBFR_ADMIN_BP"],
        "anomaly": [
            ("EFFR_ADMIN_BP", 1), ("OBFR_ADMIN_BP", 1), ("EFFR_VOLUME", -1),
        ],
    },
    "secured_repo": {
        "absolute": ["SOFR_ADMIN_BP", "TGCR_ADMIN_BP", "SOFR_TGCR_BP"],
        "anomaly": [
            ("SOFR_ADMIN_BP", 1), ("TGCR_ADMIN_BP", 1),
            ("SOFR_TGCR_BP", 1), ("SOFR_VOLUME", -1),
        ],
    },
    "dispersion_depth": {
        "absolute": ["SOFR_IQR_BP", "SOFR_UPPER_TAIL_BP", "TGCR_IQR_BP"],
        "anomaly": [
            ("SOFR_IQR_BP", 1), ("SOFR_UPPER_TAIL_BP", 1),
            ("TGCR_IQR_BP", 1), ("SOFR_VOLUME", -1),
        ],
    },
    "system_liquidity": {
        "absolute": [
            "WRESBAL_21D_CHG_PCT_BANK_ASSETS",
            "WTREGEN_21D_CHG_PCT_BANK_ASSETS",
            "PRIMARY_CREDIT_PCT_BANK_ASSETS",
            "BTFP_PCT_BANK_ASSETS",
        ],
        "anomaly": [
            ("WRESBAL_21D_CHG_PCT_BANK_ASSETS", -1),
            ("WTREGEN_21D_CHG_PCT_BANK_ASSETS", 1),
            ("PRIMARY_CREDIT_PCT_BANK_ASSETS", 1),
            ("BTFP_PCT_BANK_ASSETS", 1),
        ],
    },
    "repo_intermediation": {
        "absolute": [],
        "anomaly": [
            ("repo_dvp_overnight_volume", -1),
            ("repo_gcf_overnight_volume", -1),
            ("repo_triparty_overnight_volume", -1),
            ("dealer_fails_to_deliver", 1),
            ("dealer_fails_to_receive", 1),
        ],
    },
    "mmf": {
        "absolute": [],
        "anomaly": [
            ("mmf_total_investments_3m_change_pct", 1),
            ("mmf_repo_share_3m_change_pp", 1),
            ("mmf_bank_related_share_3m_change_pp", -1),
        ],
    },
    "cp_cd": {
        "absolute": [
            "A2P2_AA_1M_BP", "A2P2_AA_3M_BP",
            "AA_FIN_CP_3M_TSY_BP", "AA_NONFIN_CP_3M_TSY_BP",
            "FINCP_21D_CHG_PCT", "NFINCP_21D_CHG_PCT",
            "LTDACBW027NBOG_21D_CHG_PCT",
        ],
        "anomaly": [
            ("A2P2_AA_1M_BP", 1), ("A2P2_AA_3M_BP", 1),
            ("AA_FIN_CP_3M_TSY_BP", 1), ("AA_NONFIN_CP_3M_TSY_BP", 1),
            ("FINCP_21D_CHG_PCT", -1), ("NFINCP_21D_CHG_PCT", -1),
            ("LTDACBW027NBOG_21D_CHG_PCT", -1),
        ],
    },
}

SYSTEM_FACILITY_STRESS_COLUMNS = [
    "PRIMARY_CREDIT_PCT_BANK_ASSETS",
    "BTFP_PCT_BANK_ASSETS",
]


def _piecewise_risk(series: pd.Series, knots: list[list[float]]) -> pd.Series:
    if not knots:
        return pd.Series(index=series.index, dtype=float)
    ordered = sorted((float(x), float(score)) for x, score in knots)
    values = pd.to_numeric(series, errors="coerce")
    result = np.interp(
        values.to_numpy(dtype=float),
        [item[0] for item in ordered],
        [item[1] for item in ordered],
    )
    return pd.Series(result, index=series.index).where(values.notna()).clip(0, 100)


def _percentile_risk(
    series: pd.Series,
    direction: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    percentile = values.rolling(window, min_periods=min_periods).rank(pct=True)
    if direction < 0:
        percentile = 1.0 - percentile
    return (100.0 * percentile).where(values.notna()).clip(0, 100)


def _mean_available(pieces: list[pd.Series], index: pd.Index) -> pd.Series:
    if not pieces:
        return pd.Series(index=index, dtype=float)
    return pd.concat(pieces, axis=1).mean(axis=1, skipna=True)


def _hybrid(
    absolute: pd.Series,
    anomaly: pd.Series,
    absolute_weight: float,
    anomaly_weight: float,
) -> pd.Series:
    weighted = absolute.fillna(0) * absolute_weight + anomaly.fillna(0) * anomaly_weight
    available_weight = (
        absolute.notna().astype(float) * absolute_weight
        + anomaly.notna().astype(float) * anomaly_weight
    )
    return (weighted / available_weight.replace(0, np.nan)).clip(0, 100)


def calculate_scores(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if features.empty:
        return features

    weights = cfg["weights"]
    absolute_weight = float(cfg.get("absolute_weight", 0.7))
    anomaly_weight = float(cfg.get("anomaly_weight", 0.3))
    anomaly_window = int(cfg.get("anomaly_window_days", cfg.get("rolling_window_days", 1260)))
    anomaly_minimum = int(
        cfg.get("anomaly_minimum_history_days", cfg.get("minimum_history_days", 252))
    )
    thresholds = cfg.get("absolute_thresholds", {})
    out = pd.DataFrame(index=features.index)

    for component, specs in COMPONENT_SPECS.items():
        absolute_pieces = [
            _piecewise_risk(features[col], thresholds[col])
            for col in specs["absolute"]
            if col in features and col in thresholds
        ]
        anomaly_pieces = [
            _percentile_risk(features[col], direction, anomaly_window, anomaly_minimum)
            for col, direction in specs["anomaly"]
            if col in features
        ]
        absolute = _mean_available(absolute_pieces, features.index)
        if component == "system_liquidity" and cfg.get("system_facility_stress_override", True):
            facility_pieces = [
                _piecewise_risk(features[col], thresholds[col])
                for col in SYSTEM_FACILITY_STRESS_COLUMNS
                if col in features and col in thresholds
            ]
            facility_stress = _mean_available(facility_pieces, features.index)
            absolute = pd.concat([absolute, facility_stress], axis=1).max(
                axis=1, skipna=True
            )
        anomaly = _mean_available(anomaly_pieces, features.index)
        out[f"{component}_absolute"] = absolute
        out[f"{component}_anomaly"] = anomaly
        out[component] = _hybrid(absolute, anomaly, absolute_weight, anomaly_weight)

    weight_sum = pd.Series(0.0, index=out.index)
    weighted = pd.Series(0.0, index=out.index)
    for name, weight in weights.items():
        available = out[name].notna()
        weighted = weighted.add(out[name].fillna(0) * weight, fill_value=0)
        weight_sum = weight_sum.add(available.astype(float) * weight, fill_value=0)
    out["liquidity_risk_score"] = (weighted / weight_sum.replace(0, np.nan)).clip(0, 100)
    component_columns = [name for name in weights if name in out]
    out["peak_component_score"] = out[component_columns].max(axis=1, skipna=True)

    for suffix in ("absolute", "anomaly"):
        numerator = pd.Series(0.0, index=out.index)
        denominator = pd.Series(0.0, index=out.index)
        for name, weight in weights.items():
            col = f"{name}_{suffix}"
            available = out[col].notna()
            numerator = numerator.add(out[col].fillna(0) * weight, fill_value=0)
            denominator = denominator.add(available.astype(float) * weight, fill_value=0)
        out[f"liquidity_risk_score_{suffix}"] = (
            numerator / denominator.replace(0, np.nan)
        ).clip(0, 100)

    regime_thresholds = cfg.get("regime_thresholds", [25, 40, 55, 70])
    if len(regime_thresholds) != 4:
        raise ValueError("scoring.regime_thresholds must contain four ascending values")
    out["liquidity_risk_score_raw"] = out["liquidity_risk_score"]
    confirmed_components = [
        component
        for component in cfg.get("confirmed_stress_components", [])
        if component in out
    ]
    if confirmed_components:
        if len(confirmed_components) < 2:
            raise ValueError("confirmed_stress_components must contain at least two components")
        confirmation_score = out[confirmed_components].min(axis=1, skipna=False)
        watch_threshold = float(regime_thresholds[1])
        floor_strength = float(cfg.get("confirmed_stress_floor_strength", 0.25))
        confirmed_floor = watch_threshold + floor_strength * (
            confirmation_score - watch_threshold
        )
        out["confirmed_stress_floor"] = confirmed_floor.where(
            confirmation_score >= watch_threshold
        )
        out["liquidity_risk_score"] = pd.concat(
            [out["liquidity_risk_score"], out["confirmed_stress_floor"]], axis=1
        ).max(axis=1, skipna=True).clip(0, 100)
    else:
        out["confirmed_stress_floor"] = np.nan
    bins = [-np.inf, *[float(value) for value in regime_thresholds], np.inf]
    out["regime"] = pd.cut(
        out["liquidity_risk_score"],
        bins=bins,
        labels=["Ample", "Normal", "Watch", "Stressed", "Severe"],
    ).astype("string")
    out["peak_component_regime"] = pd.cut(
        out["peak_component_score"],
        bins=bins,
        labels=["Ample", "Normal", "Watch", "Stressed", "Severe"],
    ).astype("string")
    out["stressed_component_count"] = out[component_columns].ge(bins[3]).sum(axis=1)
    out["coverage"] = weight_sum / sum(weights.values())
    return out


def calculate_score_details(
    features: pd.DataFrame,
    cfg: dict,
    at: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Return the raw value and absolute/anomaly score for each model input."""
    columns = [
        "component", "variable", "raw_value", "absolute_score",
        "anomaly_score", "risk_direction",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)

    timestamp = features.index[-1] if at is None else pd.Timestamp(at)
    if timestamp not in features.index:
        raise KeyError(f"Score-detail date is not present in features: {timestamp}")

    anomaly_window = int(cfg.get("anomaly_window_days", cfg.get("rolling_window_days", 1260)))
    anomaly_minimum = int(
        cfg.get("anomaly_minimum_history_days", cfg.get("minimum_history_days", 252))
    )
    thresholds = cfg.get("absolute_thresholds", {})
    rows: list[dict] = []
    for component, specs in COMPONENT_SPECS.items():
        anomaly_directions = dict(specs["anomaly"])
        variables = list(dict.fromkeys([*specs["absolute"], *anomaly_directions]))
        for variable in variables:
            if variable not in features:
                continue
            absolute_score = np.nan
            if variable in thresholds and variable in specs["absolute"]:
                absolute_score = _piecewise_risk(
                    features[variable], thresholds[variable]
                ).loc[timestamp]
            anomaly_score = np.nan
            direction = anomaly_directions.get(variable)
            if direction is not None:
                anomaly_score = _percentile_risk(
                    features[variable], direction, anomaly_window, anomaly_minimum
                ).loc[timestamp]
            if direction is None:
                risk_direction = "fixed thresholds"
            elif direction < 0:
                risk_direction = "lower is riskier"
            else:
                risk_direction = "higher is riskier"
            rows.append({
                "component": component,
                "variable": variable,
                "raw_value": pd.to_numeric(features[variable], errors="coerce").loc[timestamp],
                "absolute_score": absolute_score,
                "anomaly_score": anomaly_score,
                "risk_direction": risk_direction,
            })
    return pd.DataFrame(rows, columns=columns)

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from funding_dashboard.series_rules import match_catalog, choose_best_match


RATE_IDS = ["EFFR", "OBFR", "TGCR", "BGCR", "SOFR"]
FNYR_VOLUME_IDS = {rate: f"FNYR-{rate}_UV-A" for rate in RATE_IDS}


def robust_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.rolling(window, min_periods=min_periods).median()
    mad = (s - median).abs().rolling(window, min_periods=min_periods).median()
    denom = 1.4826 * mad.replace(0, np.nan)
    z = (s - median) / denom
    fallback_std = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return z.fillna((s - s.rolling(window, min_periods=min_periods).mean()) / fallback_std)


def _pivot(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last").sort_index()
    all_days = pd.date_range(wide.index.min(), wide.index.max(), freq="B")
    return wide.reindex(all_days)


def _ffill_limited(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    daily_prefixes = (
        "EFFR_", "OBFR_", "TGCR_", "BGCR_", "SOFR_", "DGS", "DTB", "DCP",
        "RIFSP", "REPO-", "repo_",
    )
    for col in out.columns:
        if col.startswith(daily_prefixes) or col in {"IORB", "IOER"}:
            out[col] = out[col].ffill(limit=5)
        elif col == "TLAACBW027SBOG":
            out[col] = out[col].ffill(limit=15)
        elif col in {
            "WRESBAL", "WTREGEN", "WALCL", "LTDACBW027NBOG", "FINCP", "NFINCP",
            "WLCFLPCL", "H41RESPPALDKNWW",
        }:
            out[col] = out[col].ffill(limit=10)
        else:
            out[col] = out[col].ffill(limit=35)
    return out


def attach_rule_series(wide: pd.DataFrame, observations: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    catalog = (
        observations.groupby(["source", "dataset", "series_id"], as_index=False)
        .agg(series_name=("series_name", "last"), first_date=("date", "min"), last_date=("date", "max"), observations=("value", "count"))
    )
    selected: dict[str, dict] = {}
    out = wide.copy()
    for alias, rule in rules.items():
        matches = match_catalog(catalog, rule)
        if rule.get("mnemonics"):
            catalog_by_id = {
                str(row["series_id"]).casefold(): row
                for _, row in matches.iterrows()
            }
            ordered_matches = [
                catalog_by_id[str(series_id).casefold()]
                for series_id in rule["mnemonics"]
                if str(series_id).casefold() in catalog_by_id
            ]
            if not ordered_matches:
                selected[alias] = {"status": "not_matched"}
                continue
            combined = pd.Series(index=out.index, dtype=float)
            matched_ids = []
            for match in ordered_matches:
                series_id = match["series_id"]
                if series_id not in out.columns:
                    continue
                combined = combined.combine_first(out[series_id])
                matched_ids.append(series_id)
            if not matched_ids:
                selected[alias] = {"status": "not_matched"}
                continue
            out[alias] = combined
            selected[alias] = {
                "status": "stitched" if len(matched_ids) > 1 else "matched",
                "series_ids": matched_ids,
                "series_id": matched_ids[0],
                "series_name": ordered_matches[0]["series_name"],
                "dataset": ordered_matches[0]["dataset"],
                "selection_policy": "Use mnemonics in order; fill missing dates from the next series.",
            }
            continue

        match = choose_best_match(matches)
        if match is None:
            selected[alias] = {"status": "not_matched"}
            continue
        sid = match["series_id"]
        if sid in out.columns:
            out[alias] = out[sid]
            selected[alias] = {
                "status": "matched",
                "series_id": sid,
                "series_name": match["series_name"],
                "dataset": match["dataset"],
            }
    return out, selected


def build_features(observations: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    wide = _pivot(observations)
    if wide.empty:
        return wide, {}
    wide, selected = attach_rule_series(wide, observations, rules)
    wide = _ffill_limited(wide)

    # OFR's FNYR series provide the same official underlying reference-rate
    # volumes in dollars. Use them to fill historical gaps in the New York Fed
    # API series, whose canonical unit is USD billions.
    for rate, fallback_id in FNYR_VOLUME_IDS.items():
        if fallback_id not in wide:
            continue
        fallback = wide[fallback_id] / 1_000_000_000.0
        canonical = f"{rate}_VOLUME"
        wide[canonical] = wide[canonical].combine_first(fallback) if canonical in wide else fallback

    admin_rate = pd.Series(np.nan, index=wide.index, dtype=float)
    if "IORB" in wide:
        admin_rate = wide["IORB"]
    if "IOER" in wide:
        admin_rate = admin_rate.combine_first(wide["IOER"])
    wide["RESERVE_ADMIN_RATE"] = admin_rate

    def spread(lhs: str, rhs: str, name: str):
        if lhs in wide and rhs in wide:
            wide[name] = (wide[lhs] - wide[rhs]) * 100.0  # percent -> basis points

    for rate in RATE_IDS:
        spread(f"{rate}_RATE", "IORB", f"{rate}_IORB_BP")
        spread(f"{rate}_RATE", "RESERVE_ADMIN_RATE", f"{rate}_ADMIN_BP")
        spread(f"{rate}_RATE", "EFFR_TARGET_HIGH", f"{rate}_TARGET_HIGH_BP")
    spread("OBFR_RATE", "EFFR_RATE", "OBFR_EFFR_BP")
    spread("SOFR_RATE", "TGCR_RATE", "SOFR_TGCR_BP")
    spread("BGCR_RATE", "TGCR_RATE", "BGCR_TGCR_BP")
    spread("SOFR_RATE", "EFFR_RATE", "SOFR_EFFR_BP")

    for rate in RATE_IDS:
        p25, p75, p99 = f"{rate}_P25", f"{rate}_P75", f"{rate}_P99"
        if p25 in wide and p75 in wide:
            wide[f"{rate}_IQR_BP"] = (wide[p75] - wide[p25]) * 100
        if f"{rate}_RATE" in wide and p99 in wide:
            wide[f"{rate}_UPPER_TAIL_BP"] = (wide[p99] - wide[f"{rate}_RATE"]) * 100
        vol = f"{rate}_VOLUME"
        if vol in wide:
            wide[f"{rate}_VOLUME_Z"] = robust_zscore(wide[vol])
            wide[f"{rate}_VOLUME_5D_CHG_PCT"] = wide[vol].pct_change(5) * 100

    for col in ["WTREGEN", "WRESBAL", "RRPONTSYD", "WALCL", "FINCP", "NFINCP", "LTDACBW027NBOG"]:
        if col in wide:
            wide[f"{col}_5D_CHG"] = wide[col].diff(5)
            wide[f"{col}_21D_CHG"] = wide[col].diff(21)
            wide[f"{col}_21D_CHG_PCT"] = wide[col].pct_change(21, fill_method=None) * 100
            wide[f"{col}_Z"] = robust_zscore(wide[col])

    if {"WALCL", "WTREGEN", "RRPONTSYD"}.issubset(wide.columns):
        wide["NET_LIQUIDITY_PROXY"] = wide["WALCL"] - wide["WTREGEN"] - wide["RRPONTSYD"]
        wide["NET_LIQUIDITY_21D_CHG"] = wide["NET_LIQUIDITY_PROXY"].diff(21)

    if "TLAACBW027SBOG" in wide:
        bank_assets_millions = wide["TLAACBW027SBOG"] * 1_000.0
        for col in ["WRESBAL_21D_CHG", "WTREGEN_21D_CHG", "NET_LIQUIDITY_21D_CHG"]:
            if col in wide:
                wide[f"{col}_PCT_BANK_ASSETS"] = wide[col] / bank_assets_millions * 100
        if "WLCFLPCL" in wide:
            wide["PRIMARY_CREDIT_PCT_BANK_ASSETS"] = wide["WLCFLPCL"] / bank_assets_millions * 100
        if "H41RESPPALDKNWW" in wide:
            wide["BTFP_PCT_BANK_ASSETS"] = wide["H41RESPPALDKNWW"] / bank_assets_millions * 100

    cp_pairs = [
        ("DCPF1M", "DGS1MO", "AA_FIN_CP_1M_TSY_BP"),
        ("DCPN30", "DGS1MO", "AA_NONFIN_CP_1M_TSY_BP"),
        ("RIFSPPNA2P2D30NB", "DCPN30", "A2P2_AA_1M_BP"),
        ("DCPF3M", "DGS3MO", "AA_FIN_CP_3M_TSY_BP"),
        ("DCPN3M", "DGS3MO", "AA_NONFIN_CP_3M_TSY_BP"),
        ("RIFSPPNA2P2D90NB", "DCPN3M", "A2P2_AA_3M_BP"),
    ]
    for lhs, rhs, name in cp_pairs:
        spread(lhs, rhs, name)

    # MMF and repo proxy changes.
    for alias in rules:
        if alias not in wide:
            continue
        wide[f"{alias}_5D_CHG"] = wide[alias].diff(5)
        wide[f"{alias}_21D_CHG"] = wide[alias].diff(21)
        wide[f"{alias}_Z"] = robust_zscore(wide[alias])

    # OFR publishes aggregate MMF investments monthly, not government/prime
    # fund WAM/WAL and liquidity buckets. Build transparent allocation proxies
    # from the published totals. Sixty-three business days approximates three
    # months and avoids treating a monthly series as a daily flow.
    total_mmf = "mmf_total_investments"
    if total_mmf in wide:
        denominator = pd.to_numeric(wide[total_mmf], errors="coerce").replace(0, np.nan)
        wide["mmf_total_investments_3m_change_pct"] = (
            denominator.pct_change(63, fill_method=None) * 100
        )
        allocation_columns = {
            "repo": "mmf_repo_investments",
            "treasury": "mmf_treasury_investments",
            "bank_related": "mmf_bank_related_assets",
        }
        for label, column in allocation_columns.items():
            if column not in wide:
                continue
            share = pd.to_numeric(wide[column], errors="coerce") / denominator * 100
            wide[f"mmf_{label}_share_pct"] = share
            wide[f"mmf_{label}_share_3m_change_pp"] = share.diff(63)

    return wide, selected

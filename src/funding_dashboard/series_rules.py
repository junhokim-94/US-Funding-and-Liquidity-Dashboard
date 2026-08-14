from __future__ import annotations

import re
import pandas as pd


def _contains(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def match_catalog(catalog: pd.DataFrame, rule: dict) -> pd.DataFrame:
    work = catalog.copy()
    if "dataset" in rule:
        work = work[work["dataset"].str.casefold() == str(rule["dataset"]).casefold()]
    if rule.get("mnemonics"):
        mnemonics = {str(value).casefold() for value in rule["mnemonics"]}
        return work[work["series_id"].str.casefold().isin(mnemonics)]
    if rule.get("mnemonic"):
        return work[work["series_id"].str.casefold() == str(rule["mnemonic"]).casefold()]
    all_terms = rule.get("include_all", [])
    any_terms = rule.get("include_any", [])
    exclude = rule.get("exclude", [])

    def ok(name: str) -> bool:
        text = str(name)
        if all_terms and not all(_contains(text, t) for t in all_terms):
            return False
        if any_terms and not any(_contains(text, t) for t in any_terms):
            return False
        if exclude and any(_contains(text, t) for t in exclude):
            return False
        return True

    return work[work["series_name"].map(ok)]


def choose_best_match(matches: pd.DataFrame) -> pd.Series | None:
    if matches.empty:
        return None
    work = matches.copy()
    name = work["series_name"].str.casefold()
    score = pd.Series(0.0, index=work.index)
    score += name.str.contains("final", regex=False).astype(float) * 4
    score += name.str.contains("as of", regex=False).astype(float) * 3
    score -= name.str.contains("preliminary", regex=False).astype(float) * 5
    score += work["last_date"].rank(pct=True).fillna(0)
    work["_score"] = score
    return work.sort_values(["_score", "observations"], ascending=False).iloc[0]

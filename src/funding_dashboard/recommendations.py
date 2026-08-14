from __future__ import annotations

import pandas as pd


COMPONENT_ACTIONS = {
    "unsecured_funding": (
        "Unsecured funding",
        "Shorten the tenor of new unsecured exposures and recheck concentration by issuer and banking group. "
        "Confirm whether maturing positions can be rolled into O/N or secured transactions.",
        "Review EFFR/OBFR spreads to administered rates and underlying volumes daily until they return to normal ranges.",
    ),
    "secured_repo": (
        "Secured repo",
        "Confirm term-repo limits and collateral availability before settlement dates, and reduce reliance on any single dealer. "
        "Recalculate cash needs under higher haircuts and intraday margin calls.",
        "Review SOFR/TGCR spreads, the SOFR-TGCR basis, and repo volumes daily.",
    ),
    "dispersion_depth": (
        "Market depth / dispersion",
        "Do not infer executable levels from median rates alone; include tail rates. "
        "Split large orders and use limit orders and multiple dealer quotes.",
        "Confirm that SOFR/TGCR IQRs and upper tails decline for five consecutive business days.",
    ),
    "system_liquidity": (
        "System liquidity",
        "Set T+0 and T+7 liquidity targets at the greater of the internal minimum or scheduled settlements plus stressed outflows. "
        "Pause WAM/WAL extension and distribute maturities around tax and Treasury settlement dates.",
        "Review reserves, TGA changes, and Primary Credit/BTFP usage after each weekly H.4.1 release.",
    ),
    "repo_intermediation": (
        "Repo intermediation",
        "Confirm that dealer-level committed capacity covers the largest five-business-day funding need. "
        "Pre-position collateral and pull forward rollovers concentrated around quarter-end.",
        "Confirm that DVP, GCF, and tri-party volumes and dealer fails normalize.",
    ),
    "mmf": (
        "MMF allocation proxy",
        "Treat rapid MMF inflows and a rising repo allocation as signs of a shift into cash-like safe assets, "
        "and review bank and CP/CD funding exposure. This OFR aggregate proxy must not be used for fund-level WAM/WAL decisions.",
        "Review three-month total-investment growth, repo-share changes, and bank-related-asset share monthly.",
    ),
    "cp_cd": (
        "CP / CD",
        "When A2/P2 and Treasury-relative spreads widen, shorten new CP/CD maturities and compare the all-in cost with T-bills and repo. "
        "Recalculate issuer-level maturity concentrations and rollover needs over one-week and one-month horizons.",
        "Confirm that A2/P2-AA spreads and the contraction in CP outstanding ease for five consecutive business days.",
    ),
}


REGIME_ACTIONS = {
    "Ample": (
        "Maintain the normal policy liquidity buffer and deploy excess cash gradually.",
        "Extend WAM/WAL only within the neutral policy range; consider high-quality CP and term repo only when relative value is positive.",
        "Weekly",
    ),
    "Normal": (
        "Maintain the normal buffer after scheduled settlements and redemptions; phase new risk into the portfolio.",
        "Preserve the maturity ladder and avoid increasing single-date or issuer concentration.",
        "Twice weekly",
    ),
    "Watch": (
        "Keep T+0/T+7 available liquidity above stressed outflows plus scheduled settlements.",
        "Pause new maturity extension and add risk selectively only after the score remains below 40 for five consecutive business days.",
        "Daily",
    ),
    "Stressed": (
        "Pre-fund known settlements and hold same-day and weekly liquidity above normal policy levels.",
        "Reduce unsecured and lower-quality exposures; shift toward secured and short-dated instruments.",
        "Intraday / daily",
    ),
    "Severe": (
        "Activate the contingency funding plan under simultaneous redemption, margin-call, and settlement-failure scenarios.",
        "Stop new risk deployment; operate primarily in cash, T-bills, and O/N secured instruments, and immediately reapprove management limits.",
        "Intraday",
    ),
}


def make_recommendations(features: pd.DataFrame, scores: pd.DataFrame, rv: pd.DataFrame) -> list[str]:
    if scores.empty or scores["liquidity_risk_score"].dropna().empty:
        return ["Insufficient data to produce a portfolio recommendation."]
    latest_score = float(scores["liquidity_risk_score"].dropna().iloc[-1])
    regime = str(scores["regime"].dropna().iloc[-1]) if scores["regime"].notna().any() else "Unknown"
    recs: list[str] = [f"Funding regime: {regime}; liquidity risk score: {latest_score:.1f}/100."]

    if regime == "Severe":
        recs += [
            "Raise same-day and weekly liquidity buffers; shorten WAM/WAL and pre-fund known settlements.",
            "Favor Treasury bills and overnight/short-term secured placements; reduce lower-tier unsecured credit and counterparty concentrations.",
            "Escalate dealer capacity, collateral, and margin-call stress tests to daily review.",
        ]
    elif regime == "Stressed":
        recs += [
            "Maintain above-normal liquidity, shorten incremental purchases, and stagger maturities around tax, settlement, and quarter-end dates.",
            "Add term repo only where capacity is committed; reduce reliance on a single dealer or cash-investor channel.",
        ]
    elif regime == "Watch":
        recs += [
            "Keep a neutral-to-cautious maturity profile and preserve optionality for upcoming Treasury settlements.",
            "Add spread product selectively when compensation is wide and issuer/counterparty limits remain available.",
        ]
    else:
        recs += [
            "Liquidity conditions are broadly supportive; deploy excess cash gradually while retaining normal redemption and settlement buffers.",
            "Consider measured extension into term repo or high-quality CP when spread pickup exceeds historical norms.",
        ]

    if not rv.empty:
        wide = rv[rv["relative_value_signal"] >= 1.5]
        tight = rv[rv["relative_value_signal"] <= -1.5]
        if not wide.empty:
            recs.append("Potentially attractive compensation: " + ", ".join(wide["instrument_or_spread"].head(3)) + ".")
        if not tight.empty:
            recs.append("Relatively rich/tight areas: " + ", ".join(tight["instrument_or_spread"].head(3)) + ".")
    recs.append("Recommendations are rule-based research outputs, not investment advice; apply fund-specific limits, liquidity rules, and credit approvals.")
    return recs


def make_detailed_action_plan(scores: pd.DataFrame) -> pd.DataFrame:
    columns = ["Priority", "Scope", "Specific action", "Exit / validation condition"]
    if scores.empty or scores["liquidity_risk_score"].dropna().empty:
        return pd.DataFrame(columns=columns)

    latest_index = scores["liquidity_risk_score"].last_valid_index()
    latest = scores.loc[latest_index]
    regime = str(latest.get("regime", "Unknown"))
    liquidity_action, risk_action, cadence = REGIME_ACTIONS.get(regime, REGIME_ACTIONS["Normal"])
    rows = [
        {
            "Priority": 1,
            "Scope": f"Whole portfolio ({regime})",
            "Specific action": liquidity_action,
            "Exit / validation condition": f"Review {cadence.lower()}; remain in the next-lower regime for five consecutive business days.",
        },
        {
            "Priority": 2,
            "Scope": "New risk and maturity",
            "Specific action": risk_action,
            "Exit / validation condition": "Both the total score and peak component are below Watch.",
        },
    ]

    component_scores = {
        component: float(latest[component])
        for component in COMPONENT_ACTIONS
        if component in latest and pd.notna(latest[component])
    }
    for priority, (component, score) in enumerate(
        sorted(component_scores.items(), key=lambda item: item[1], reverse=True)[:3],
        start=3,
    ):
        label, action, validation = COMPONENT_ACTIONS[component]
        rows.append({
            "Priority": priority,
            "Scope": f"{label} ({score:.1f})",
            "Specific action": action,
            "Exit / validation condition": validation,
        })
    return pd.DataFrame(rows, columns=columns)


def regime_playbook() -> pd.DataFrame:
    ranges = {
        "Ample": "0-25",
        "Normal": ">25-40",
        "Watch": ">40-55",
        "Stressed": ">55-70",
        "Severe": ">70-100",
    }
    rows = []
    for regime, (liquidity, risk, cadence) in REGIME_ACTIONS.items():
        rows.append({
            "Band": f"{regime} ({ranges[regime]})",
            "Liquidity operations": liquidity,
            "New risk / maturity": risk,
            "Review cadence": cadence,
        })
    return pd.DataFrame(rows)

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from funding_dashboard.dashboard_data import load_s3_dashboard_data
from funding_dashboard.features import build_features
from funding_dashboard.recommendations import (
    make_detailed_action_plan,
    make_recommendations,
    regime_playbook,
)
from funding_dashboard.relative_value import build_rv_table
from funding_dashboard.scoring import calculate_score_details, calculate_scores
from funding_dashboard.settings import load_settings
from funding_dashboard.storage import Storage


COMPONENT_LABELS = {
    "unsecured_funding": "Unsecured funding",
    "secured_repo": "Secured repo",
    "dispersion_depth": "Dispersion / depth",
    "system_liquidity": "System liquidity",
    "repo_intermediation": "Repo intermediation",
    "mmf": "MMF allocation proxy",
    "cp_cd": "CP / CD",
}

VARIABLE_META = {
    "EFFR_ADMIN_BP": ("EFFR − reserve admin rate", "bp"),
    "OBFR_ADMIN_BP": ("OBFR − reserve admin rate", "bp"),
    "EFFR_VOLUME": ("EFFR underlying volume", "USD bn"),
    "SOFR_ADMIN_BP": ("SOFR − reserve admin rate", "bp"),
    "TGCR_ADMIN_BP": ("TGCR − reserve admin rate", "bp"),
    "SOFR_TGCR_BP": ("SOFR − TGCR", "bp"),
    "SOFR_VOLUME": ("SOFR underlying volume", "USD bn"),
    "SOFR_IQR_BP": ("SOFR 25th–75th percentile range", "bp"),
    "SOFR_UPPER_TAIL_BP": ("SOFR 99th percentile − median", "bp"),
    "TGCR_IQR_BP": ("TGCR 25th–75th percentile range", "bp"),
    "WRESBAL_21D_CHG_PCT_BANK_ASSETS": ("Reserve balance 21d change / bank assets", "%"),
    "WTREGEN_21D_CHG_PCT_BANK_ASSETS": ("TGA 21d change / bank assets", "%"),
    "PRIMARY_CREDIT_PCT_BANK_ASSETS": ("Primary Credit / bank assets", "%"),
    "BTFP_PCT_BANK_ASSETS": ("BTFP / bank assets", "%"),
    "repo_dvp_overnight_volume": ("DVP overnight repo volume", "USD"),
    "repo_gcf_overnight_volume": ("GCF overnight repo volume", "USD"),
    "repo_triparty_overnight_volume": ("Tri-party overnight repo volume", "USD"),
    "dealer_fails_to_deliver": ("Primary dealer fails to deliver", "USD mn"),
    "dealer_fails_to_receive": ("Primary dealer fails to receive", "USD mn"),
    "mmf_total_investments_3m_change_pct": ("MMF total investments 3m change", "%"),
    "mmf_repo_share_3m_change_pp": ("MMF repo share 3m change", "pp"),
    "mmf_bank_related_share_3m_change_pp": ("MMF bank-related share 3m change", "pp"),
    "A2P2_AA_1M_BP": ("A2/P2 − AA nonfinancial CP, 1m", "bp"),
    "A2P2_AA_3M_BP": ("A2/P2 − AA nonfinancial CP, 3m", "bp"),
    "AA_FIN_CP_3M_TSY_BP": ("AA financial CP − Treasury, 3m", "bp"),
    "AA_NONFIN_CP_3M_TSY_BP": ("AA nonfinancial CP − Treasury, 3m", "bp"),
    "FINCP_21D_CHG_PCT": ("Financial CP outstanding 21d change", "%"),
    "NFINCP_21D_CHG_PCT": ("Nonfinancial CP outstanding 21d change", "%"),
    "LTDACBW027NBOG_21D_CHG_PCT": ("Large time deposits 21d change", "%"),
}

HISTORICAL_EVENTS = {
    "2019 Repo stress": ("2019-09-16", "2019-09-18"),
    "2020 COVID funding stress": ("2020-03-09", "2020-03-31"),
    "2023 Banking stress": ("2023-03-10", "2023-03-31"),
}

REPO_VENUE_LABELS = {
    "repo_dvp_overnight_volume": "DVP overnight transaction volume",
    "repo_gcf_overnight_volume": "GCF overnight transaction volume",
    "repo_triparty_overnight_volume": "Tri-party overnight transaction volume",
}

CP_RATE_GROUPS = {
    "30-day CP and 1-month Treasury rates": {
        "DCPF1M": "AA financial CP (30d)",
        "DCPN30": "AA nonfinancial CP (30d)",
        "RIFSPPNA2P2D30NB": "A2/P2 nonfinancial CP (30d)",
        "DGS1MO": "U.S. Treasury (1m)",
    },
    "90-day CP and 3-month Treasury rates": {
        "DCPF3M": "AA financial CP (90d)",
        "DCPN3M": "AA nonfinancial CP (90d)",
        "RIFSPPNA2P2D90NB": "A2/P2 nonfinancial CP (90d)",
        "DGS3MO": "U.S. Treasury (3m)",
    },
}

CP_SPREAD_LABELS = {
    "AA_FIN_CP_1M_TSY_BP": "AA financial CP (30d) - Treasury (1m)",
    "AA_NONFIN_CP_1M_TSY_BP": "AA nonfinancial CP (30d) - Treasury (1m)",
    "A2P2_AA_1M_BP": "A2/P2 - AA nonfinancial CP (30d)",
    "AA_FIN_CP_3M_TSY_BP": "AA financial CP (90d) - Treasury (3m)",
    "AA_NONFIN_CP_3M_TSY_BP": "AA nonfinancial CP (90d) - Treasury (3m)",
    "A2P2_AA_3M_BP": "A2/P2 - AA nonfinancial CP (90d)",
}

CP_BALANCE_LABELS = {
    "FINCP": "Financial CP outstanding",
    "NFINCP": "Nonfinancial CP outstanding",
    "LTDACBW027NBOG": "Large time deposits at commercial banks",
}


def dashboard_data_source() -> str:
    source = os.getenv("DASHBOARD_DATA_SOURCE", "local").strip().lower()
    if source not in {"local", "s3"}:
        raise ValueError(
            "DASHBOARD_DATA_SOURCE must be 'local' or 's3' "
            f"(received {source!r})"
        )
    return source


@st.cache_data(ttl=900)
def load_all(config_path: str, data_source: str):
    settings = load_settings(config_path)
    if data_source == "s3":
        features, scores, catalog, rv, recs, auctions, selected = load_s3_dashboard_data(settings)
        return settings, features, scores, rv, recs, catalog, auctions, selected

    storage = Storage(settings.db_path, read_only=True)
    obs = storage.observations()
    catalog = storage.catalog()
    auctions = storage.auctions((pd.Timestamp.today() - pd.Timedelta(days=30)).date().isoformat())
    storage.close()
    features, selected = build_features(obs, settings.config.get("ofr_rules", {}))
    scores = calculate_scores(features, settings.config["scoring"])
    rv = build_rv_table(features)
    recs = make_recommendations(features, scores, rv)
    return settings, features, scores, rv, recs, catalog, auctions, selected


def _regime(score: float, thresholds: list[float]) -> str:
    for threshold, label in zip(thresholds, ["Ample", "Normal", "Watch", "Stressed"]):
        if score <= threshold:
            return label
    return "Severe"


def _component_summary(scores: pd.DataFrame, cfg: dict, at: pd.Timestamp) -> pd.DataFrame:
    latest = scores.loc[at]
    weights = cfg["weights"]
    available_weight = sum(
        weight for component, weight in weights.items() if pd.notna(latest.get(component))
    )
    thresholds = [float(value) for value in cfg["regime_thresholds"]]
    rows = []
    for component, weight in weights.items():
        score = latest.get(component)
        effective_weight = weight / available_weight if pd.notna(score) and available_weight else None
        rows.append({
            "Component": COMPONENT_LABELS.get(component, component),
            "Final score": round(float(score), 1) if pd.notna(score) else None,
            "Absolute score": round(float(latest.get(f"{component}_absolute")), 1)
            if pd.notna(latest.get(f"{component}_absolute")) else None,
            "Anomaly score": round(float(latest.get(f"{component}_anomaly")), 1)
            if pd.notna(latest.get(f"{component}_anomaly")) else None,
            "Model weight": f"{weight:.0%}",
            "Effective weight": f"{effective_weight:.1%}" if effective_weight is not None else "0.0%",
            "Score contribution": round(float(score) * effective_weight, 1)
            if effective_weight is not None else None,
            "Regime": _regime(float(score), thresholds) if pd.notna(score) else "Unavailable",
        })
    return pd.DataFrame(rows)


def _variable_score_table(features: pd.DataFrame, cfg: dict, at: pd.Timestamp) -> pd.DataFrame:
    details = calculate_score_details(features, cfg, at).copy()
    details["Variable"] = details["variable"].map(
        lambda value: VARIABLE_META.get(value, (value, ""))[0]
    )
    details["Unit"] = details["variable"].map(
        lambda value: VARIABLE_META.get(value, (value, ""))[1]
    )
    details["Component"] = details["component"].map(COMPONENT_LABELS)
    details["Risk direction"] = details["risk_direction"].map({
        "higher is riskier": "Higher = more stress",
        "lower is riskier": "Lower = more stress",
        "fixed thresholds": "Fixed thresholds",
    })
    for column in ["raw_value", "absolute_score", "anomaly_score"]:
        details[column] = pd.to_numeric(details[column], errors="coerce").round(2)
    return details[[
        "Component", "Variable", "raw_value", "Unit",
        "absolute_score", "anomaly_score", "Risk direction",
    ]].rename(columns={
        "raw_value": "Latest value",
        "absolute_score": "Absolute score",
        "anomaly_score": "Anomaly score",
    })


def _evidence_value(
    features: pd.DataFrame,
    start: str,
    end: str,
    column: str,
    operation: str,
    label: str,
    unit: str,
) -> str | None:
    if column not in features:
        return None
    values = pd.to_numeric(features.loc[start:end, column], errors="coerce").dropna()
    if values.empty:
        return None
    value = values.min() if operation == "min" else values.max()
    return f"{label} {value:.1f}{unit}"


def _historical_event_table(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    evidence_specs = {
        "2019 Repo stress": [
            ("SOFR_ADMIN_BP", "max", "SOFR-admin max", "bp"),
            ("SOFR_VOLUME", "max", "SOFR volume max", "bn"),
        ],
        "2020 COVID funding stress": [
            ("A2P2_AA_1M_BP", "max", "A2/P2-AA 1m max", "bp"),
            ("FINCP_21D_CHG_PCT", "min", "Financial CP 21d min", "%"),
        ],
        "2023 Banking stress": [
            ("PRIMARY_CREDIT_PCT_BANK_ASSETS", "max", "Primary Credit max", "%"),
            ("BTFP_PCT_BANK_ASSETS", "max", "BTFP max", "%"),
            ("mmf_total_investments_3m_change_pct", "max", "MMF 3m inflow max", "%"),
        ],
    }
    component_columns = [component for component in weights if component in scores]
    rows = []
    for event, (start, end) in HISTORICAL_EVENTS.items():
        window = scores.loc[start:end]
        total = window["liquidity_risk_score"].dropna()
        if total.empty:
            continue
        component_peaks = window[component_columns].max().sort_values(ascending=False)
        evidence = [
            _evidence_value(features, start, end, column, operation, label, unit)
            for column, operation, label, unit in evidence_specs[event]
        ]
        rows.append({
            "Episode": event,
            "Window": f"{start} – {end}",
            "Mean score": round(float(total.mean()), 1),
            "Max score": round(float(total.max()), 1),
            "Peak date": total.idxmax().date().isoformat(),
            "Peak component": COMPONENT_LABELS.get(component_peaks.index[0], component_peaks.index[0]),
            "Peak component score": round(float(component_peaks.iloc[0]), 1),
            "Observed market evidence": "; ".join(value for value in evidence if value),
        })
    return pd.DataFrame(rows)


def _add_event_windows(figure):
    short_labels = ["2019 repo", "2020 COVID", "2023 banks"]
    vertical_positions = [0.98, 0.88, 0.98]
    for (label, (start, end)), short_label, y_position in zip(
        HISTORICAL_EVENTS.items(), short_labels, vertical_positions
    ):
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(214, 39, 40, 0.08)",
            line_width=0,
        )
        midpoint = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        figure.add_annotation(
            x=midpoint,
            y=y_position,
            yref="paper",
            text=short_label,
            showarrow=False,
            font={"size": 10},
        )
    return figure


def _render_action_plan(action_plan: pd.DataFrame) -> None:
    for row in action_plan.to_dict(orient="records"):
        with st.container(border=True):
            st.markdown(f"**{row['Priority']}. {row['Scope']}**")
            st.write(row["Specific action"])
            st.caption(f"Exit / validation condition: {row['Exit / validation condition']}")


def _render_historical_events(historical: pd.DataFrame) -> None:
    columns = st.columns(len(historical))
    for column, row in zip(columns, historical.to_dict(orient="records")):
        with column:
            with st.container(border=True):
                st.markdown(f"**{row['Episode']}**")
                st.caption(row["Window"])
                st.metric("Maximum score", f"{row['Max score']:.1f}")
                st.write(f"Mean score: **{row['Mean score']:.1f}**")
                st.write(f"Peak date: **{row['Peak date']}**")
                st.write(
                    f"Peak component: **{row['Peak component']} "
                    f"({row['Peak component score']:.1f})**"
                )
                st.caption(row["Observed market evidence"])


def _repo_vintage_table(catalog: pd.DataFrame, selected: dict) -> pd.DataFrame:
    rows = []
    for alias, label in REPO_VENUE_LABELS.items():
        series_ids = selected.get(alias, {}).get("series_ids", [])
        for series_id in series_ids:
            match = catalog[catalog["series_id"] == series_id]
            if match.empty:
                continue
            item = match.sort_values("last_date").iloc[-1]
            is_final = str(series_id).upper().endswith("-F")
            rows.append({
                "Market": label.replace(" overnight transaction volume", ""),
                "Vintage": "Final" if is_final else "Preliminary",
                "Mnemonic": series_id,
                "Last source observation": str(pd.Timestamp(item["last_date"]).date()),
                "Usage": "Preferred when available" if is_final else "Extends beyond the final-series cutoff",
            })
    return pd.DataFrame(rows)


def _cp_rate_catalog_table(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in CP_RATE_GROUPS.values():
        for series_id, label in group.items():
            match = catalog[
                catalog["series_id"].eq(series_id) & catalog["source"].eq("fred")
            ]
            rows.append({
                "Display label": label,
                "FRED series ID": series_id,
                "Latest source observation": (
                    str(pd.Timestamp(match.iloc[-1]["last_date"]).date())
                    if not match.empty else "Unavailable"
                ),
            })
    return pd.DataFrame(rows)


def _line_chart(
    frame: pd.DataFrame,
    title: str,
    event_windows: bool = False,
    y_label: str = "Value",
    legend_title: str = "Variable",
):
    plot_frame = frame.dropna(how="all").copy()
    plot_frame.index.name = "Date"
    plot_frame.columns.name = None
    # Streamlit renders every tab eagerly. Plotly's automatic ScatterGL mode
    # can therefore exhaust the browser's WebGL-context limit and blank the
    # charts created first. SVG keeps the full history without that failure.
    figure = px.line(plot_frame, title=title, render_mode="svg")
    figure.update_layout(hovermode="x unified", legend_title_text=legend_title)
    figure.update_xaxes(title_text="Date", rangeslider_visible=True)
    figure.update_yaxes(title_text=y_label)
    figure.update_traces(connectgaps=False)
    if event_windows:
        _add_event_windows(figure)
    return figure


def main():
    st.set_page_config(page_title="Funding & Liquidity Dashboard", layout="wide")
    config_path = str(Path(__file__).resolve().parents[2] / "config.yml")
    try:
        settings, features, scores, rv, recs, catalog, auctions, selected = load_all(
            config_path, dashboard_data_source()
        )
    except Exception as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        st.stop()
    st.title("U.S. Funding & Liquidity Dashboard")
    st.caption("Official APIs only: New York Fed, OFR, FRED/Federal Reserve, and U.S. Treasury Fiscal Data.")

    if features.empty:
        st.warning("No data yet. Run: python run_update.py")
        return

    latest_date = features.dropna(how="all").index.max()
    score_date = scores["liquidity_risk_score"].last_valid_index()
    latest_score = float(scores.at[score_date, "liquidity_risk_score"])
    latest_regime = str(scores.at[score_date, "regime"])
    latest_peak = float(scores.at[score_date, "peak_component_score"])
    coverage = float(scores.at[score_date, "coverage"])

    # Keep the date outside the narrow metric row so it cannot be ellipsized.
    st.markdown(f"**As of: {latest_date.strftime('%Y-%m-%d')}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Liquidity risk", f"{latest_score:.1f}")
    c2.metric("Regime", latest_regime)
    c3.metric("Peak component", f"{latest_peak:.1f}")
    c4.metric("Score coverage", f"{coverage:.0%}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Rates & Spreads", "Liquidity Score", "Repo & MMF", "CP / CD", "Relative Value", "Data Quality"
    ])

    with tab1:
        rate_cols = [c for c in ["EFFR_RATE", "OBFR_RATE", "TGCR_RATE", "BGCR_RATE", "SOFR_RATE", "IORB"] if c in features]
        rate_labels = {column: column.replace("_RATE", "") for column in rate_cols}
        st.plotly_chart(
            _line_chart(
                features[rate_cols].rename(columns=rate_labels),
                "Overnight reference rates",
                y_label="Rate (%)",
                legend_title="Rate",
            ),
            use_container_width=True,
        )
        spread_cols = [c for c in ["EFFR_IORB_BP", "OBFR_IORB_BP", "TGCR_IORB_BP", "BGCR_IORB_BP", "SOFR_IORB_BP", "SOFR_TGCR_BP"] if c in features]
        spread_labels = {
            column: column.replace("_IORB_BP", " - IORB").replace("_TGCR_BP", " - TGCR")
            for column in spread_cols
        }
        st.plotly_chart(
            _line_chart(
                features[spread_cols].rename(columns=spread_labels),
                "Funding spreads",
                y_label="Spread (bp)",
                legend_title="Spread",
            ),
            use_container_width=True,
        )
        vol_cols = [c for c in ["EFFR_VOLUME", "OBFR_VOLUME", "TGCR_VOLUME", "BGCR_VOLUME", "SOFR_VOLUME"] if c in features]
        volume_labels = {column: column.replace("_VOLUME", "") for column in vol_cols}
        st.plotly_chart(
            _line_chart(
                features[vol_cols].rename(columns=volume_labels),
                "Reference-rate underlying volumes",
                y_label="Volume (USD bn)",
                legend_title="Rate",
            ),
            use_container_width=True,
        )

    with tab2:
        component_cols = [c for c in settings.config["scoring"]["weights"] if c in scores]
        score_history = scores[["liquidity_risk_score"] + component_cols]
        score_figure = _line_chart(
            score_history,
            "Liquidity risk score and components — full available history",
            event_windows=True,
        )
        for threshold in settings.config["scoring"]["regime_thresholds"]:
            score_figure.add_hline(y=float(threshold), line_dash="dot", line_width=1, opacity=0.35)
        st.plotly_chart(score_figure, use_container_width=True)
        st.caption(
            "Hybrid score: 70% fixed absolute thresholds and 30% trailing five-year percentile anomaly. "
            "The system-liquidity component preserves concentrated Primary Credit/BTFP stress. When system liquidity and CP/CD both reach Watch, "
            "a small confirmation floor prevents cross-market banking stress from being averaged away. MMF is an OFR aggregate allocation proxy."
        )

        st.subheader("Component score detail")
        st.dataframe(
            _component_summary(scores, settings.config["scoring"], score_date),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Underlying variable score detail")
        st.dataframe(
            _variable_score_table(features, settings.config["scoring"], score_date),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Absolute score is mapped to fixed market thresholds. Anomaly score is the variable's percentile within up to five years of prior observations; "
            "a minimum of 252 observations is required."
        )

        st.subheader("Portfolio actions")
        for rec in recs:
            st.write("•", rec)

        st.markdown("#### Detailed current action plan")
        _render_action_plan(make_detailed_action_plan(scores))

        with st.expander("Score-band operating playbook", expanded=False):
            st.dataframe(regime_playbook(), use_container_width=True, hide_index=True)
            st.caption(
                "Operational examples must be calibrated to the portfolio's mandate, regulatory liquidity rules, counterparty limits, and redemption profile."
            )

        st.subheader("Historical liquidity-stress examples")
        historical = _historical_event_table(
            features, scores, settings.config["scoring"]["weights"]
        )
        _render_historical_events(historical)
        st.caption(
            "Backtest windows are diagnostic event studies, not estimates of future loss. The revised model is expected to reach at least "
            "Stressed in 2019/2020 and Watch in the 2023 banking episode while limiting 2024 Watch false positives."
        )

    with tab3:
        repo_cols = [column for column in REPO_VENUE_LABELS if column in features]
        if repo_cols:
            venue_volumes = (
                features[repo_cols]
                .div(1_000_000_000_000.0)
                .rename(columns=REPO_VENUE_LABELS)
            )
            st.plotly_chart(
                _line_chart(venue_volumes, "OFR overnight repo transaction volumes (USD tn)"),
                use_container_width=True,
            )
            st.caption(
                "The series uses OFR final observations when available and preliminary daily observations after the final-series cutoff. "
                "Up to five business days are forward-filled for release timing; longer source gaps remain visible."
            )
            vintage_table = _repo_vintage_table(catalog, selected)
            if not vintage_table.empty:
                with st.expander("Repo data-vintage status", expanded=False):
                    st.dataframe(vintage_table, use_container_width=True, hide_index=True)

        additional_repo_cols = [column for column in [
            "repo_term_volume",
            "dealer_repo",
            "dealer_reverse_repo",
            "dealer_fails_to_deliver",
            "dealer_fails_to_receive",
            "sponsored_repo_volume",
            "sponsored_reverse_repo_volume",
        ] if column in features]
        if additional_repo_cols:
            st.plotly_chart(
                _line_chart(features[additional_repo_cols], "Additional repo intermediation proxies (native source units)"),
                use_container_width=True,
            )

        if "mmf_total_investments" in features:
            total = pd.DataFrame({
                "MMF total investments (USD tn)": features["mmf_total_investments"] / 1_000_000_000_000,
            })
            st.plotly_chart(_line_chart(total, "OFR MMF total investments"), use_container_width=True)
        mmf_share_cols = [c for c in [
            "mmf_repo_share_pct", "mmf_treasury_share_pct", "mmf_bank_related_share_pct"
        ] if c in features]
        if mmf_share_cols:
            st.plotly_chart(_line_chart(features[mmf_share_cols], "OFR MMF allocation shares (%)"), use_container_width=True)
        mmf_stress_cols = [c for c in [
            "mmf_total_investments_3m_change_pct",
            "mmf_repo_share_3m_change_pp",
            "mmf_bank_related_share_3m_change_pp",
        ] if c in features]
        if mmf_stress_cols:
            st.plotly_chart(_line_chart(features[mmf_stress_cols], "MMF allocation stress inputs"), use_container_width=True)
        st.caption(
            "OFR MMF data are monthly aggregate investments. They do not replace SEC Form N-MFP fund-level WAM/WAL or daily/weekly liquidity fields."
        )
        if not auctions.empty:
            st.subheader("Recent and upcoming Treasury auctions")
            st.dataframe(auctions.sort_values("auction_date").tail(30), use_container_width=True, hide_index=True)

    with tab4:
        for title, labels in CP_RATE_GROUPS.items():
            cp_cols = [column for column in labels if column in features]
            if not cp_cols:
                continue
            cp_rates = features[cp_cols].rename(columns=labels)
            st.plotly_chart(
                _line_chart(
                    cp_rates,
                    title,
                    y_label="Rate (%)",
                    legend_title="Instrument",
                ),
                use_container_width=True,
            )
        st.caption(
            "A gap in a CP line means the Federal Reserve reported no rate because eligible trade data were insufficient. "
            "Values are carried forward for at most five business days; longer gaps remain visible and are not treated as zero."
        )
        with st.expander("CP rate labels and source freshness", expanded=False):
            st.dataframe(
                _cp_rate_catalog_table(catalog),
                use_container_width=True,
                hide_index=True,
            )
        spread_cols = [column for column in CP_SPREAD_LABELS if column in features]
        if spread_cols:
            spreads = features[spread_cols].rename(columns=CP_SPREAD_LABELS)
            st.plotly_chart(
                _line_chart(
                    spreads,
                    "CP credit and liquidity spreads",
                    y_label="Spread (bp)",
                    legend_title="Spread",
                ),
                use_container_width=True,
            )
        proxy_cols = [column for column in CP_BALANCE_LABELS if column in features]
        if proxy_cols:
            balances = features[proxy_cols].rename(columns=CP_BALANCE_LABELS)
            st.plotly_chart(
                _line_chart(
                    balances,
                    "CP outstanding and large-time-deposit proxy",
                    y_label="USD billions",
                    legend_title="Series",
                ),
                use_container_width=True,
            )

    with tab5:
        st.dataframe(rv, use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("Selected OFR series")
        st.json(selected)
        st.subheader("Available series catalog")
        st.dataframe(catalog, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

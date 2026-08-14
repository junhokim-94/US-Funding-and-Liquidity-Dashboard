from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from funding_dashboard.settings import Settings
from funding_dashboard.storage import Storage
from funding_dashboard.features import build_features
from funding_dashboard.scoring import calculate_scores
from funding_dashboard.relative_value import build_rv_table
from funding_dashboard.recommendations import make_recommendations


def _write_source_exports(
    observations: pd.DataFrame,
    auctions: pd.DataFrame,
    output_dir,
) -> dict[str, str]:
    """Write normalized, validated source extracts for the S3 raw layer."""
    source_masks = {
        "raw_nyfed.parquet": observations["source"].eq("nyfed"),
        "raw_fred.parquet": observations["source"].eq("fred"),
        "raw_ofr.parquet": observations["source"].isin(["ofr", "ofr_hf"]),
    }
    written: dict[str, str] = {}
    for filename, mask in source_masks.items():
        path = output_dir / filename
        observations.loc[mask].to_parquet(path, index=False)
        written[filename] = str(path)

    treasury_path = output_dir / "raw_treasury.parquet"
    auctions.to_parquet(treasury_path, index=False)
    written[treasury_path.name] = str(treasury_path)
    return written


def run_analytics(settings: Settings) -> dict:
    with Storage(settings.db_path) as storage:
        obs = storage.observations()
        catalog = storage.catalog()
        auctions = storage.auctions()

    features, selected = build_features(obs, settings.config.get("ofr_rules", {}))
    if features.empty:
        raise RuntimeError("Analytics produced no features")
    scores = calculate_scores(features, settings.config["scoring"])
    rv = build_rv_table(features)
    recommendations = make_recommendations(features, scores, rv)

    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source_exports = _write_source_exports(obs, auctions, out_dir)
    features.to_parquet(out_dir / "features.parquet")
    scores.to_parquet(out_dir / "scores.parquet")
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    rv.to_csv(out_dir / "relative_value.csv", index=False)
    auctions.to_csv(out_dir / "treasury_auctions.csv", index=False)
    (out_dir / "recommendations.json").write_text(
        json.dumps(recommendations, indent=2), encoding="utf-8"
    )
    (out_dir / "selected_ofr_series.json").write_text(
        json.dumps(selected, indent=2, default=str), encoding="utf-8"
    )
    return {
        "features": features,
        "scores": scores,
        "catalog": catalog,
        "relative_value": rv,
        "auctions": auctions,
        "recommendations": recommendations,
        "selected_series": selected,
        "source_exports": source_exports,
        "output_files": sorted(str(path) for path in out_dir.iterdir() if path.is_file()),
    }

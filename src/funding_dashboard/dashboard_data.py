"""Read a committed dashboard snapshot from S3 without downloading DuckDB."""

from __future__ import annotations

from io import BytesIO
import json
from typing import Any

import boto3
import pandas as pd

from funding_dashboard.settings import Settings


_REQUIRED_ARTIFACTS = {
    "features.parquet",
    "scores.parquet",
    "catalog.parquet",
    "relative_value.csv",
    "treasury_auctions.csv",
    "recommendations.json",
    "selected_ofr_series.json",
}


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _read_bytes(client: Any, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _read_json(client: Any, bucket: str, key: str) -> dict:
    return json.loads(_read_bytes(client, bucket, key))


def load_s3_dashboard_data(
    settings: Settings,
    *,
    client: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, dict]:
    """Load artifacts from the single atomically promoted S3 pointer."""
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET must be set when DASHBOARD_DATA_SOURCE=s3")

    client = client or boto3.client("s3", region_name=settings.aws_region)
    latest_key = _join_key(settings.s3_prefix, "latest.json")
    latest = _read_json(client, settings.s3_bucket, latest_key)
    if latest.get("status") != "success":
        raise RuntimeError("S3 latest.json does not reference a successful run")

    artifacts = latest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("S3 latest.json has no artifact map")
    missing = sorted(_REQUIRED_ARTIFACTS - artifacts.keys())
    if missing:
        raise RuntimeError("S3 latest.json is missing required artifacts: " + ", ".join(missing))

    def read_parquet(name: str) -> pd.DataFrame:
        return pd.read_parquet(BytesIO(_read_bytes(client, settings.s3_bucket, artifacts[name])))

    def read_csv(name: str) -> pd.DataFrame:
        return pd.read_csv(BytesIO(_read_bytes(client, settings.s3_bucket, artifacts[name])))

    features = read_parquet("features.parquet")
    scores = read_parquet("scores.parquet")
    catalog = read_parquet("catalog.parquet")
    rv = read_csv("relative_value.csv")
    auctions = read_csv("treasury_auctions.csv")
    recommendations = _read_json(client, settings.s3_bucket, artifacts["recommendations.json"])
    selected = _read_json(client, settings.s3_bucket, artifacts["selected_ofr_series.json"])

    if not isinstance(recommendations, list) or not isinstance(selected, dict):
        raise RuntimeError("S3 dashboard JSON artifacts have unexpected formats")
    return features, scores, catalog, rv, recommendations, auctions, selected

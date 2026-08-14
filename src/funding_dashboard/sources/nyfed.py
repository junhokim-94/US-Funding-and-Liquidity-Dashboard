from __future__ import annotations

from datetime import date, timedelta
import json
import logging

import pandas as pd

from funding_dashboard.http import HttpClient

LOGGER = logging.getLogger(__name__)
BASE = "https://markets.newyorkfed.org/api/rates"


def _date_chunks(start: date, end: date, max_days: int):
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=max_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_reference_rates(client: HttpClient, start_date: str, end_date: str, max_days: int = 365) -> pd.DataFrame:
    rows: list[dict] = []
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for chunk_start, chunk_end in _date_chunks(start, end, max_days):
        response = client.get(
            f"{BASE}/all/search.json",
            params={
                "startDate": chunk_start.isoformat(),
                "endDate": chunk_end.isoformat(),
            },
            expire_after=6 * 3600,
        )
        payload = response.json()
        for item in payload.get("refRates", []):
            rate_type = str(item.get("type", "")).upper()
            if rate_type not in {"EFFR", "OBFR", "TGCR", "BGCR", "SOFR"}:
                continue
            effective = item.get("effectiveDate")
            fields = {
                "RATE": ("percentRate", "%"),
                "P01": ("percentPercentile1", "%"),
                "P25": ("percentPercentile25", "%"),
                "P75": ("percentPercentile75", "%"),
                "P99": ("percentPercentile99", "%"),
                "VOLUME": ("volumeInBillions", "USD bn"),
                "TARGET_LOW": ("targetRateFrom", "%"),
                "TARGET_HIGH": ("targetRateTo", "%"),
            }
            for suffix, (json_key, unit) in fields.items():
                value = item.get(json_key)
                if value is None:
                    continue
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.isna(numeric_value):
                    LOGGER.debug(
                        "Skipping non-numeric NY Fed field date=%s type=%s field=%s value=%r",
                        effective,
                        rate_type,
                        json_key,
                        value,
                    )
                    continue
                rows.append({
                    "source": "nyfed",
                    "dataset": "reference_rates",
                    "series_id": f"{rate_type}_{suffix}",
                    "series_name": f"{rate_type} {suffix.replace('_', ' ').title()}",
                    "date": effective,
                    "value": float(numeric_value),
                    "unit": unit,
                    "metadata_json": json.dumps({
                        "revisionIndicator": item.get("revisionIndicator", ""),
                        "raw_type": rate_type,
                    }),
                })
    return pd.DataFrame(rows)

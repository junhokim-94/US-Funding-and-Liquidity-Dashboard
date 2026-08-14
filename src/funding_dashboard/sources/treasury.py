from __future__ import annotations

from datetime import date, timedelta
import json

import pandas as pd

from funding_dashboard.http import HttpClient

URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
FIELDS = [
    "record_date", "auction_date", "issue_date", "maturity_date", "cusip",
    "security_type", "security_term", "offering_amt", "total_accepted",
    "bid_to_cover_ratio", "high_yield"
]


def fetch_auctions(client: HttpClient, days_back: int = 90, days_forward: int = 45) -> pd.DataFrame:
    start = (date.today() - timedelta(days=days_back)).isoformat()
    end = (date.today() + timedelta(days=days_forward)).isoformat()
    params = {
        "fields": ",".join(FIELDS),
        "filter": f"auction_date:gte:{start},auction_date:lte:{end}",
        "page[size]": 1000,
        "sort": "auction_date",
    }
    response = client.get(URL, params=params, expire_after=6 * 3600)
    data = response.json().get("data", [])
    rows = []
    for item in data:
        row = {k: item.get(k) for k in FIELDS}
        row["raw_json"] = json.dumps(item)
        rows.append(row)
    return pd.DataFrame(rows)

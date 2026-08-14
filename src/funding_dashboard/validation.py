from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterable

import pandas as pd


OBS_REQUIRED_COLUMNS = (
    "source",
    "dataset",
    "series_id",
    "series_name",
    "date",
    "value",
)
OBS_KEY_COLUMNS = ("source", "dataset", "series_id", "date")
AUCTION_REQUIRED_COLUMNS = (
    "auction_date",
    "cusip",
    "security_type",
    "security_term",
)


class DataValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    name: str
    rows: int
    duplicate_rows: int = 0
    missing_required_values: int = 0
    invalid_dates: int = 0
    invalid_numeric_values: int = 0
    out_of_range_rates: int = 0
    stale_series: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return asdict(self) | {"ok": self.ok}

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DataValidationError(f"{self.name}: " + "; ".join(self.errors))


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


def validate_observations(frame: pd.DataFrame, cfg: dict, name: str = "observations") -> ValidationReport:
    report = ValidationReport(name=name, rows=len(frame))
    if frame.empty:
        report.errors.append("no rows returned")
        return report

    missing_columns = _missing_columns(frame, OBS_REQUIRED_COLUMNS)
    if missing_columns:
        report.errors.append(f"missing columns: {', '.join(missing_columns)}")
        return report

    work = frame.copy()
    missing_mask = work[list(OBS_REQUIRED_COLUMNS)].isna().any(axis=1)
    report.missing_required_values = int(missing_mask.sum())

    parsed_dates = pd.to_datetime(work["date"], errors="coerce")
    report.invalid_dates = int(parsed_dates.isna().sum())
    numeric_values = pd.to_numeric(work["value"], errors="coerce")
    report.invalid_numeric_values = int(numeric_values.isna().sum())

    report.duplicate_rows = int(work.duplicated(list(OBS_KEY_COLUMNS), keep=False).sum())

    unit = work.get("unit", pd.Series(index=work.index, dtype="object")).fillna("").astype(str).str.lower()
    series_id = work["series_id"].fillna("").astype(str).str.upper()
    is_rate = unit.str.contains("%|percent", regex=True) | series_id.str.endswith(
        ("_RATE", "_P01", "_P25", "_P75", "_P99", "_TARGET_LOW", "_TARGET_HIGH")
    )
    rate_min = float(cfg.get("rate_min_percent", -5.0))
    rate_max = float(cfg.get("rate_max_percent", 30.0))
    out_of_range = is_rate & numeric_values.notna() & ~numeric_values.between(rate_min, rate_max)
    report.out_of_range_rates = int(out_of_range.sum())

    if cfg.get("fail_on_missing_required_values", True) and report.missing_required_values:
        report.errors.append(f"{report.missing_required_values} rows have missing required values")
    if report.invalid_dates:
        report.errors.append(f"{report.invalid_dates} rows have invalid dates")
    if report.invalid_numeric_values:
        report.errors.append(f"{report.invalid_numeric_values} rows have invalid numeric values")
    if cfg.get("fail_on_duplicate_rows", True) and report.duplicate_rows:
        report.errors.append(f"{report.duplicate_rows} duplicate-key rows detected")
    if report.out_of_range_rates:
        report.errors.append(
            f"{report.out_of_range_rates} rate observations outside [{rate_min}, {rate_max}] percent"
        )
    return report


def validate_auctions(frame: pd.DataFrame, name: str = "treasury_auctions") -> ValidationReport:
    report = ValidationReport(name=name, rows=len(frame))
    if frame.empty:
        report.errors.append("no rows returned")
        return report
    missing_columns = _missing_columns(frame, AUCTION_REQUIRED_COLUMNS)
    if missing_columns:
        report.errors.append(f"missing columns: {', '.join(missing_columns)}")
        return report
    missing_mask = frame[list(AUCTION_REQUIRED_COLUMNS)].isna().any(axis=1)
    report.missing_required_values = int(missing_mask.sum())
    parsed_dates = pd.to_datetime(frame["auction_date"], errors="coerce")
    report.invalid_dates = int(parsed_dates.isna().sum())
    report.duplicate_rows = int(frame.duplicated(["cusip", "auction_date"], keep=False).sum())
    if report.missing_required_values:
        report.errors.append(f"{report.missing_required_values} rows have missing required values")
    if report.invalid_dates:
        report.errors.append(f"{report.invalid_dates} rows have invalid auction dates")
    if report.duplicate_rows:
        report.errors.append(f"{report.duplicate_rows} duplicate auction-key rows detected")
    return report


def assess_staleness(frame: pd.DataFrame, cfg: dict, as_of: date | None = None) -> ValidationReport:
    report = ValidationReport(name="staleness", rows=len(frame))
    if frame.empty:
        report.errors.append("cannot assess staleness on an empty dataset")
        return report

    as_of_ts = pd.Timestamp(as_of or date.today()).normalize()
    work = frame[["source", "dataset", "series_id", "date"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])

    daily_days = int(cfg.get("stale_daily_days", 10))
    weekly_days = int(cfg.get("stale_weekly_days", 21))
    monthly_days = int(cfg.get("stale_monthly_days", 62))
    critical = set(cfg.get("critical_series", []))

    for keys, group in work.groupby(["source", "dataset", "series_id"], sort=False):
        dates = group["date"].drop_duplicates().sort_values()
        if dates.empty:
            continue
        gaps = dates.diff().dt.days.dropna()
        median_gap = float(gaps.median()) if not gaps.empty else 1.0
        if median_gap <= 3:
            threshold = daily_days
            frequency = "daily"
        elif median_gap <= 10:
            threshold = weekly_days
            frequency = "weekly"
        else:
            threshold = monthly_days
            frequency = "monthly"
        last_date = dates.iloc[-1].normalize()
        age_days = int((as_of_ts - last_date).days)
        if age_days > threshold:
            source, dataset, series_id = keys
            item = {
                "source": source,
                "dataset": dataset,
                "series_id": series_id,
                "last_date": last_date.date().isoformat(),
                "age_days": age_days,
                "threshold_days": threshold,
                "inferred_frequency": frequency,
                "critical": series_id in critical,
            }
            report.stale_series.append(item)

    critical_stale = [item for item in report.stale_series if item["critical"]]
    fail_on_any = bool(cfg.get("fail_on_any_stale", False))
    fail_on_critical = bool(cfg.get("fail_on_critical_stale", False))

    if report.stale_series:
        message = f"{len(report.stale_series)} stale series detected"
        if fail_on_any:
            report.errors.append(message)
        else:
            report.warnings.append(message)
    if critical_stale:
        message = f"{len(critical_stale)} critical series are stale"
        if fail_on_critical and not fail_on_any:
            report.errors.append(message)
        elif not fail_on_any:
            report.warnings.append(message)
    return report

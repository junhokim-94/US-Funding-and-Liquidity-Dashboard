from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_run_metadata(metadata: dict[str, Any], metadata_root: Path) -> Path:
    run_date = str(metadata["run_date"])
    run_id = str(metadata["run_id"])
    target_dir = metadata_root / f"date={run_date}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"run_status_{run_id}.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    temp.replace(target)
    return target

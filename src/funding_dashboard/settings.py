from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


def _resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Settings:
    root: Path
    config: dict
    db_override: Path | None = None
    output_override: Path | None = None

    def _project_path(self, config_key: str, env_key: str) -> Path:
        raw = os.getenv(env_key) or self.config["project"][config_key]
        return _resolve_path(self.root, raw)

    @property
    def db_path(self) -> Path:
        return self.db_override or self._project_path("database", "DATABASE_PATH")

    @property
    def cache_dir(self) -> Path:
        return self._project_path("cache_dir", "CACHE_DIR")

    @property
    def output_dir(self) -> Path:
        return self.output_override or self._project_path("output_dir", "OUTPUT_DIR")

    @property
    def staging_dir(self) -> Path:
        return self._project_path("staging_dir", "STAGING_DIR")

    @property
    def run_metadata_dir(self) -> Path:
        return self._project_path("run_metadata_dir", "RUN_METADATA_DIR")

    @property
    def start_date(self) -> str:
        return os.getenv("START_DATE", self.config["project"]["start_date"])

    @property
    def output_mode(self) -> str:
        return os.getenv(
            "OUTPUT_MODE",
            self.config.get("publication", {}).get("output_mode", "local"),
        ).strip().lower()

    @property
    def s3_bucket(self) -> str | None:
        return os.getenv("S3_BUCKET") or None

    @property
    def s3_prefix(self) -> str:
        return (os.getenv("S3_PREFIX") or self.config.get("publication", {}).get("s3_prefix", "")).strip("/")

    @property
    def aws_region(self) -> str:
        return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    def validate_runtime(self) -> None:
        valid_modes = {"local", "s3", "both"}
        if self.output_mode not in valid_modes:
            raise ValueError(
                f"Unsupported OUTPUT_MODE={self.output_mode!r}; expected one of {sorted(valid_modes)}"
            )
        if self.output_mode in {"s3", "both"} and not self.s3_bucket:
            raise ValueError("S3_BUCKET must be set when OUTPUT_MODE is s3 or both")

    def with_project_paths(self, *, database: Path, output_dir: Path) -> "Settings":
        return Settings(
            root=self.root,
            config=deepcopy(self.config),
            db_override=database,
            output_override=output_dir,
        )


def load_settings(config_path: str | Path = "config.yml") -> Settings:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    load_dotenv(root / ".env")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    settings = Settings(root=root, config=cfg)
    for path in [
        settings.db_path.parent,
        settings.cache_dir,
        settings.output_dir.parent,
        settings.staging_dir,
        settings.run_metadata_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return settings


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

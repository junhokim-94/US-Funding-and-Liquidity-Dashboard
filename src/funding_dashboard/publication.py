from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
import uuid

import boto3

from funding_dashboard.settings import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3Artifact:
    local_path: Path
    staging_key: str
    final_key: str
    content_type: str

    @property
    def size_bytes(self) -> int:
        return self.local_path.stat().st_size


def atomic_replace_file(staged_file: Path, final_file: Path) -> None:
    final_file.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_file, final_file)


def atomic_replace_directory(staged_dir: Path, final_dir: Path) -> None:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = final_dir.with_name(f"{final_dir.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if final_dir.exists():
        final_dir.replace(backup)
    try:
        staged_dir.replace(final_dir)
    except Exception:
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if backup.exists():
            backup.replace(final_dir)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def atomic_publish_run(
    *,
    staged_db: Path | None,
    final_db: Path | None,
    staged_outputs: Path,
    final_outputs: Path,
) -> None:
    """Publish database and outputs as one rollback-capable local commit."""
    db_backup = final_db.with_suffix(final_db.suffix + ".backup") if final_db else None
    outputs_backup = final_outputs.with_name(final_outputs.name + ".backup")
    db_published = False
    outputs_published = False

    for backup in [db_backup, outputs_backup]:
        if backup and backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()

    try:
        if final_db and staged_db:
            final_db.parent.mkdir(parents=True, exist_ok=True)
            if final_db.exists():
                final_db.replace(db_backup)
        final_outputs.parent.mkdir(parents=True, exist_ok=True)
        if final_outputs.exists():
            final_outputs.replace(outputs_backup)

        if final_db and staged_db:
            staged_db.replace(final_db)
            db_published = True
        staged_outputs.replace(final_outputs)
        outputs_published = True
    except Exception:
        if outputs_published and final_outputs.exists():
            shutil.rmtree(final_outputs)
        if db_published and final_db and final_db.exists():
            final_db.unlink()
        if db_backup and db_backup.exists() and final_db:
            db_backup.replace(final_db)
        if outputs_backup.exists():
            outputs_backup.replace(final_outputs)
        raise
    else:
        if db_backup and db_backup.exists():
            db_backup.unlink()
        if outputs_backup.exists():
            shutil.rmtree(outputs_backup)


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _output_key_map(run_date: str, run_id: str) -> dict[str, str]:
    """Map local artifacts to immutable, run-scoped S3 keys.

    S3 does not provide a transaction across multiple objects. The run_id segment
    prevents a failed retry from overwriting files belonging to an earlier
    successful run on the same date. The run-status object is copied last and is
    the commit marker for the immutable object set.
    """
    run_segment = f"run_id={run_id}"
    return {
        "raw_nyfed.parquet": (
            f"raw/source=nyfed/date={run_date}/{run_segment}/data.parquet"
        ),
        "raw_fred.parquet": (
            f"raw/source=fred/date={run_date}/{run_segment}/data.parquet"
        ),
        "raw_ofr.parquet": (
            f"raw/source=ofr/date={run_date}/{run_segment}/data.parquet"
        ),
        "raw_treasury.parquet": (
            f"raw/source=treasury/date={run_date}/{run_segment}/data.parquet"
        ),
        "features.parquet": (
            f"processed/date={run_date}/{run_segment}/funding_data.parquet"
        ),
        "scores.parquet": (
            f"metrics/date={run_date}/{run_segment}/liquidity_metrics.parquet"
        ),
        "catalog.parquet": (
            f"catalog/date={run_date}/{run_segment}/series_catalog.parquet"
        ),
        "relative_value.csv": (
            f"reports/date={run_date}/{run_segment}/relative_value.csv"
        ),
        "treasury_auctions.csv": (
            f"reports/date={run_date}/{run_segment}/treasury_auctions.csv"
        ),
        "recommendations.json": (
            f"reports/date={run_date}/{run_segment}/recommendations.json"
        ),
        "selected_ofr_series.json": (
            f"reports/date={run_date}/{run_segment}/selected_ofr_series.json"
        ),
        "validation_report.json": (
            f"reports/date={run_date}/{run_segment}/validation_report.json"
        ),
    }


def _database_checkpoint_key(run_date: str, run_id: str) -> str:
    return f"state/date={run_date}/run_id={run_id}/funding.duckdb"


def _latest_key(settings: Settings) -> str:
    return _join_key(settings.s3_prefix, "latest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    return guessed or "application/octet-stream"


def build_s3_publication_plan(
    settings: Settings,
    output_dir: Path,
    run_id: str,
    run_date: str | None = None,
    database_path: Path | None = None,
) -> tuple[list[S3Artifact], str, str]:
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET must be set when OUTPUT_MODE is s3 or both")

    run_date = run_date or date.today().isoformat()
    prefix = settings.s3_prefix
    staging_root = _join_key(prefix, "_staging", f"run_id={run_id}")
    artifacts: list[S3Artifact] = []

    missing: list[str] = []
    for filename, final_relative_key in _output_key_map(run_date, run_id).items():
        local_path = output_dir / filename
        if not local_path.is_file():
            missing.append(filename)
            continue
        artifacts.append(
            S3Artifact(
                local_path=local_path,
                staging_key=_join_key(staging_root, final_relative_key),
                final_key=_join_key(prefix, final_relative_key),
                content_type=_content_type(local_path),
            )
        )

    if database_path is not None:
        if not database_path.is_file():
            missing.append(str(database_path))
        else:
            checkpoint_relative_key = _database_checkpoint_key(run_date, run_id)
            artifacts.append(
                S3Artifact(
                    local_path=database_path,
                    staging_key=_join_key(staging_root, checkpoint_relative_key),
                    final_key=_join_key(prefix, checkpoint_relative_key),
                    content_type="application/vnd.duckdb",
                )
            )

    if missing:
        raise FileNotFoundError(
            "Required S3 publication artifacts are missing: " + ", ".join(sorted(missing))
        )

    metadata_relative_key = f"run_metadata/date={run_date}/run_status_{run_id}.json"
    metadata_staging_key = _join_key(staging_root, metadata_relative_key)
    metadata_final_key = _join_key(prefix, metadata_relative_key)
    return artifacts, metadata_staging_key, metadata_final_key


def planned_s3_uris(
    settings: Settings,
    output_dir: Path,
    run_id: str,
    run_date: str | None = None,
    database_path: Path | None = None,
) -> list[str]:
    artifacts, _, metadata_final_key = build_s3_publication_plan(
        settings, output_dir, run_id, run_date, database_path
    )
    return [
        *(f"s3://{settings.s3_bucket}/{artifact.final_key}" for artifact in artifacts),
        f"s3://{settings.s3_bucket}/{metadata_final_key}",
    ]


def _head_and_verify(client: Any, bucket: str, key: str, expected_size: int) -> None:
    response = client.head_object(Bucket=bucket, Key=key)
    actual_size = int(response.get("ContentLength", -1))
    if actual_size != expected_size:
        raise RuntimeError(
            f"S3 object size mismatch for s3://{bucket}/{key}: "
            f"expected={expected_size} actual={actual_size}"
        )


def _delete_keys_best_effort(client: Any, bucket: str, keys: list[str]) -> None:
    if not keys:
        return
    try:
        for start in range(0, len(keys), 1000):
            batch = keys[start : start + 1000]
            client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )
    except Exception as exc:  # cleanup must not invalidate a completed commit
        LOGGER.warning("S3 cleanup failed bucket=%s keys=%s error=%s", bucket, len(keys), exc)


def _s3_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error", {})
    return str(error.get("Code")) if error.get("Code") is not None else None


def _read_latest_pointer(
    client: Any,
    bucket: str,
    key: str,
) -> tuple[dict | None, str | None]:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _s3_error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None, None
        raise

    body = response["Body"]
    raw = body.read() if hasattr(body, "read") else body
    try:
        pointer = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid latest pointer s3://{bucket}/{key}") from exc
    if not isinstance(pointer, dict):
        raise RuntimeError(f"Invalid latest pointer s3://{bucket}/{key}: expected object")
    return pointer, response.get("ETag")


def _pointer_order(pointer: dict) -> tuple[str, str]:
    return (
        str(pointer.get("started_at_utc") or ""),
        str(pointer.get("run_id") or ""),
    )


def _build_latest_pointer(
    metadata: dict,
    artifacts: list[S3Artifact],
    metadata_final_key: str,
) -> dict:
    if metadata.get("status") != "success":
        raise ValueError("latest.json can only reference a successful run")

    artifact_keys = {
        artifact.local_path.name: artifact.final_key for artifact in artifacts
    }
    artifact_sizes = {
        artifact.local_path.name: artifact.size_bytes for artifact in artifacts
    }
    checkpoint = next(
        (artifact for artifact in artifacts if artifact.final_key.endswith("/funding.duckdb")),
        None,
    )
    checkpoint_payload = None
    if checkpoint is not None:
        checkpoint_payload = {
            "key": checkpoint.final_key,
            "size_bytes": checkpoint.size_bytes,
            "sha256": _sha256_file(checkpoint.local_path),
        }

    return {
        "schema_version": 1,
        "status": "success",
        "run_id": str(metadata.get("run_id") or ""),
        "run_date": str(metadata.get("run_date") or ""),
        "started_at_utc": str(metadata.get("started_at_utc") or ""),
        "finished_at_utc": str(metadata.get("finished_at_utc") or ""),
        "metadata_key": metadata_final_key,
        "artifacts": artifact_keys,
        "artifact_sizes": artifact_sizes,
        "database_checkpoint": checkpoint_payload,
    }


def _promote_latest_pointer(
    client: Any,
    bucket: str,
    key: str,
    candidate: dict,
    *,
    max_attempts: int = 5,
) -> bool:
    """Atomically promote a committed run without allowing an older run to win."""
    body = json.dumps(candidate, indent=2, sort_keys=True).encode("utf-8")
    for _ in range(max_attempts):
        current, etag = _read_latest_pointer(client, bucket, key)
        if current is not None and _pointer_order(current) >= _pointer_order(candidate):
            LOGGER.info(
                "Skipped latest pointer promotion candidate_run_id=%s current_run_id=%s",
                candidate.get("run_id"),
                current.get("run_id"),
            )
            return False

        conditions = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                CacheControl="no-cache",
                ServerSideEncryption="AES256",
                **conditions,
            )
            return True
        except Exception as exc:
            if _s3_error_code(exc) in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
                continue

            # A timeout can happen after S3 accepted the conditional write. Read
            # the pointer once before surfacing an ambiguous failure.
            try:
                observed, _ = _read_latest_pointer(client, bucket, key)
            except Exception:
                raise exc
            if observed is not None and observed.get("run_id") == candidate.get("run_id"):
                return True
            if observed is not None and _pointer_order(observed) > _pointer_order(candidate):
                return False
            raise

    current, _ = _read_latest_pointer(client, bucket, key)
    if current is not None and current.get("run_id") == candidate.get("run_id"):
        return True
    if current is not None and _pointer_order(current) > _pointer_order(candidate):
        return False
    raise RuntimeError(f"Could not atomically update s3://{bucket}/{key}")


def restore_latest_database_checkpoint(
    settings: Settings,
    target_path: Path,
    *,
    client: Any | None = None,
) -> dict | None:
    """Restore the latest committed DuckDB checkpoint into a run-local path."""
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET must be set to restore a database checkpoint")

    bucket = str(settings.s3_bucket)
    client = client or boto3.client("s3", region_name=settings.aws_region)
    pointer_key = _latest_key(settings)
    pointer, _ = _read_latest_pointer(client, bucket, pointer_key)
    if pointer is None:
        LOGGER.info("No latest S3 pointer found; starting without a checkpoint")
        return None

    checkpoint = pointer.get("database_checkpoint")
    if not isinstance(checkpoint, dict):
        LOGGER.info("Latest S3 run has no DuckDB checkpoint; starting without one")
        return None

    checkpoint_key = str(checkpoint.get("key") or "")
    expected_prefix = _join_key(settings.s3_prefix, "state") + "/"
    if not checkpoint_key.startswith(expected_prefix):
        raise RuntimeError(f"Unsafe DuckDB checkpoint key in latest pointer: {checkpoint_key!r}")

    expected_size = int(checkpoint.get("size_bytes", -1))
    expected_sha256 = str(checkpoint.get("sha256") or "")
    if expected_size < 0 or len(expected_sha256) != 64:
        raise RuntimeError("Latest DuckDB checkpoint is missing integrity metadata")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(
        f".{target_path.name}.{uuid.uuid4().hex}.download"
    )
    try:
        client.download_file(bucket, checkpoint_key, str(temporary_path))
        actual_size = temporary_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"DuckDB checkpoint size mismatch: expected={expected_size} actual={actual_size}"
            )
        actual_sha256 = _sha256_file(temporary_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError("DuckDB checkpoint SHA-256 mismatch")
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    LOGGER.info(
        "Restored DuckDB checkpoint run_id=%s path=%s",
        pointer.get("run_id"),
        target_path,
    )
    return pointer


def publish_outputs_to_s3(
    settings: Settings,
    output_dir: Path,
    metadata_path: Path,
    run_id: str,
    run_date: str | None = None,
    database_path: Path | None = None,
    *,
    client: Any | None = None,
) -> list[str]:
    """Publish an immutable S3 object set, then atomically promote latest.json.

    All artifacts are uploaded and size-verified under the staging prefix first.
    They are then copied to run-scoped final keys. The run-status JSON is copied
    after every immutable artifact. Only then can a conditional latest.json write
    expose the run to readers. If anything fails before the marker is committed,
    partial objects from this run are deleted and earlier runs remain untouched.
    """
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata file not found: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid run metadata file: {metadata_path}") from exc
    if metadata.get("run_id") != run_id:
        raise ValueError(
            f"Run metadata ID mismatch: expected={run_id!r} actual={metadata.get('run_id')!r}"
        )

    artifacts, metadata_staging_key, metadata_final_key = build_s3_publication_plan(
        settings, output_dir, run_id, run_date, database_path
    )
    bucket = str(settings.s3_bucket)
    client = client or boto3.client("s3", region_name=settings.aws_region)
    latest_key = _latest_key(settings)
    latest_pointer = _build_latest_pointer(metadata, artifacts, metadata_final_key)

    staged_keys: list[str] = []
    copied_final_keys: list[str] = []
    committed = False
    metadata_size = metadata_path.stat().st_size

    try:
        # Phase 1: upload and verify every data artifact under _staging.
        for artifact in artifacts:
            client.upload_file(
                str(artifact.local_path),
                bucket,
                artifact.staging_key,
                ExtraArgs={
                    "ContentType": artifact.content_type,
                    "ServerSideEncryption": "AES256",
                },
            )
            staged_keys.append(artifact.staging_key)
            _head_and_verify(client, bucket, artifact.staging_key, artifact.size_bytes)

        # Stage and verify the commit marker, but do not expose it in final yet.
        client.upload_file(
            str(metadata_path),
            bucket,
            metadata_staging_key,
            ExtraArgs={
                "ContentType": "application/json",
                "ServerSideEncryption": "AES256",
            },
        )
        staged_keys.append(metadata_staging_key)
        _head_and_verify(client, bucket, metadata_staging_key, metadata_size)

        # Phase 2: copy immutable data artifacts to final run-scoped keys.
        for artifact in artifacts:
            client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": artifact.staging_key},
                Key=artifact.final_key,
                MetadataDirective="COPY",
                ServerSideEncryption="AES256",
            )
            copied_final_keys.append(artifact.final_key)
            _head_and_verify(client, bucket, artifact.final_key, artifact.size_bytes)

        # Commit marker is deliberately the last final object.
        client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": metadata_staging_key},
            Key=metadata_final_key,
            MetadataDirective="COPY",
            ServerSideEncryption="AES256",
        )
        copied_final_keys.append(metadata_final_key)
        _head_and_verify(client, bucket, metadata_final_key, metadata_size)
        committed = True

        # latest.json is the mutable visibility pointer. Conditional writes
        # prevent an older overlapping run from replacing a newer committed run.
        _promote_latest_pointer(client, bucket, latest_key, latest_pointer)

    except Exception:
        if not committed:
            _delete_keys_best_effort(client, bucket, copied_final_keys)
        _delete_keys_best_effort(client, bucket, staged_keys)
        raise

    _delete_keys_best_effort(client, bucket, staged_keys)
    return [
        *(f"s3://{bucket}/{artifact.final_key}" for artifact in artifacts),
        f"s3://{bucket}/{metadata_final_key}",
    ]


def write_validation_report(output_dir: Path, reports: Iterable[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "validation_report.json"
    target.write_text(json.dumps(list(reports), indent=2, default=str), encoding="utf-8")
    return target

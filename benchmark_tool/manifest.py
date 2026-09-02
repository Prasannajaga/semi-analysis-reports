"""Typed root run manifest."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from benchmark_tool.config import StrictModel


JobStatus = Literal["planned", "completed", "failed", "unsupported"]
RunStatus = Literal["running", "planned", "completed", "completed-with-errors", "failed"]


class ManifestJob(StrictModel):
    job_id: str
    path: str
    status: JobStatus
    reason: str | None = None


class RunManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    run_name: str
    benchmark_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: str
    updated_at: str
    completed_at: str | None = None
    status: RunStatus = "running"
    reason: str | None = None
    dry_run: bool
    api_key_env: str
    versions: dict[str, str | None]
    matrix: dict[str, int]
    jobs: list[ManifestJob] = Field(default_factory=list)

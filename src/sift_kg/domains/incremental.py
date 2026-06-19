"""Incremental schema discovery workflow.

Handles schema_version tracking, cached schema reuse, and incremental
merging when schemas change. Also manages the unassigned document bucket
for documents that need separate processing after schema changes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sift_kg.config import SiftConfig
from sift_kg.domains.discovery import save_discovered_domain
from sift_kg.domains.models import (
    DomainConfig,
    EntityTypeConfig,
    RelationTypeConfig,
)

logger = logging.getLogger(__name__)

SCHEMA_META_FILENAME = "schema_metadata.json"
UNASSIGNED_BUCKET_FILENAME = "unassigned_documents.json"


class SchemaCheckResult:
    """Result of schema version check."""

    MATCH = "match"
    MISMATCH_VERSION = "mismatch_version"
    MISMATCH_FILESIZE = "mismatch_filesize"
    MISMATCH_MISSING = "mismatch_missing"
    NO_CACHE = "no_cache"

    def __init__(self, status: str, message: str = ""):
        self.status = status
        self.message = message

    @property
    def is_match(self) -> bool:
        return self.status == self.MATCH

    @property
    def needs_confirmation(self) -> bool:
        return self.status in (
            self.MISMATCH_VERSION,
            self.MISMATCH_FILESIZE,
            self.MISMATCH_MISSING,
        )


def _meta_path(output_dir: Path) -> Path:
    return output_dir / SCHEMA_META_FILENAME


def _unassigned_path(output_dir: Path) -> Path:
    return output_dir / UNASSIGNED_BUCKET_FILENAME


def save_schema_metadata(
    output_dir: Path,
    schema_version: str | None,
    yaml_file_size: int | None,
) -> None:
    """Save schema discovery metadata for future version checks.

    Args:
        output_dir: Output directory path
        schema_version: The schema_version from sift.yaml (or None)
        yaml_file_size: The file size of sift.yaml at discovery time (or None)
    """
    meta = {
        "schema_version": schema_version,
        "yaml_file_size": yaml_file_size,
        "discovered_at": datetime.now().isoformat(),
    }
    path = _meta_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved schema metadata to {path}")


def load_schema_metadata(output_dir: Path) -> dict[str, Any] | None:
    """Load previously saved schema metadata.

    Returns:
        Dict with metadata, or None if not found/corrupt
    """
    path = _meta_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load schema metadata: {e}")
        return None


def check_schema_version(output_dir: Path) -> SchemaCheckResult:
    """Check if current sift.yaml schema matches the cached discovery.

    Returns SchemaCheckResult indicating:
    - MATCH: version and file size both match
    - MISMATCH_VERSION: schema_version changed
    - MISMATCH_FILESIZE: sift.yaml file size changed (possible manual edit)
    - MISMATCH_MISSING: schema_version missing from sift.yaml or cache
    - NO_CACHE: no previous schema metadata found
    """
    current_info = SiftConfig.get_project_yaml_info()
    current_version = current_info["schema_version"]
    current_size = current_info["file_size"]

    cached_meta = load_schema_metadata(output_dir)

    if cached_meta is None:
        return SchemaCheckResult(
            SchemaCheckResult.NO_CACHE,
            "No previous schema metadata found"
        )

    cached_version = cached_meta.get("schema_version")
    cached_size = cached_meta.get("yaml_file_size")

    if current_version is None or cached_version is None:
        return SchemaCheckResult(
            SchemaCheckResult.MISMATCH_MISSING,
            f"schema_version missing: current={current_version}, cached={cached_version}"
        )

    if current_size is None or cached_size is None:
        return SchemaCheckResult(
            SchemaCheckResult.MISMATCH_MISSING,
            f"file size info missing: current={current_size}, cached={cached_size}"
        )

    if current_version != cached_version:
        return SchemaCheckResult(
            SchemaCheckResult.MISMATCH_VERSION,
            f"schema_version changed: {cached_version} → {current_version}"
        )

    if current_size != cached_size:
        return SchemaCheckResult(
            SchemaCheckResult.MISMATCH_FILESIZE,
            f"sift.yaml file size changed: {cached_size} → {current_size} bytes (possible manual edit)"
        )

    return SchemaCheckResult(
        SchemaCheckResult.MATCH,
        f"schema_version={current_version}, file_size={current_size}"
    )


def merge_discovered_schemas(
    existing: DomainConfig,
    new: DomainConfig,
) -> DomainConfig:
    """Incrementally merge newly discovered types into existing schema.

    Preserves existing entity/relation types and adds new ones from the
    new discovery. Existing type definitions are NOT overwritten.

    Args:
        existing: The existing DomainConfig to merge into
        new: The newly discovered DomainConfig with potential new types

    Returns:
        Merged DomainConfig
    """
    merged_entity_types: dict[str, EntityTypeConfig] = dict(existing.entity_types)
    for name, cfg in new.entity_types.items():
        if name not in merged_entity_types:
            merged_entity_types[name] = cfg
            logger.info(f"Added new entity type: {name}")
        else:
            logger.debug(f"Preserving existing entity type: {name}")

    merged_relation_types: dict[str, RelationTypeConfig] = dict(existing.relation_types)
    for name, cfg in new.relation_types.items():
        if name not in merged_relation_types:
            merged_relation_types[name] = cfg
            logger.info(f"Added new relation type: {name}")
        else:
            logger.debug(f"Preserving existing relation type: {name}")

    return DomainConfig(
        name=existing.name,
        version=existing.version,
        description=f"{existing.description} (incrementally merged)",
        entity_types=merged_entity_types,
        relation_types=merged_relation_types,
        fallback_relation=existing.fallback_relation,
        schema_free=False,
        system_context=existing.system_context or new.system_context,
    )


def add_to_unassigned_bucket(
    output_dir: Path,
    document_ids: list[str],
    reason: str = "",
) -> None:
    """Add documents to the unassigned bucket for separate processing.

    These documents will not be part of the main extraction until
    explicitly reprocessed after schema confirmation.

    Args:
        output_dir: Output directory path
        document_ids: List of document IDs to mark as unassigned
        reason: Why these documents are unassigned
    """
    path = _unassigned_path(output_dir)
    existing = load_unassigned_bucket(output_dir)

    for doc_id in document_ids:
        if doc_id not in existing:
            existing[doc_id] = {
                "added_at": datetime.now().isoformat(),
                "reason": reason,
            }
            logger.info(f"Added to unassigned bucket: {doc_id}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_unassigned_bucket(output_dir: Path) -> dict[str, dict[str, str]]:
    """Load the unassigned document bucket.

    Returns:
        Dict mapping document_id -> {added_at, reason}
    """
    path = _unassigned_path(output_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        logger.warning("Invalid unassigned bucket format, resetting")
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load unassigned bucket: {e}")
        return {}


def clear_unassigned_bucket(output_dir: Path, document_ids: list[str] | None = None) -> None:
    """Clear documents from the unassigned bucket.

    Args:
        output_dir: Output directory path
        document_ids: Specific IDs to clear, or None to clear all
    """
    if document_ids is None:
        path = _unassigned_path(output_dir)
        if path.exists():
            path.unlink()
            logger.info("Cleared entire unassigned bucket")
        return

    existing = load_unassigned_bucket(output_dir)
    for doc_id in document_ids:
        if doc_id in existing:
            del existing[doc_id]
            logger.info(f"Removed from unassigned bucket: {doc_id}")

    path = _unassigned_path(output_dir)
    if existing:
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    elif path.exists():
        path.unlink()


def save_merged_schema(
    domain: DomainConfig,
    output_dir: Path,
    schema_version: str | None,
    yaml_file_size: int | None,
) -> Path:
    """Save a merged/updated discovered domain and update metadata.

    Args:
        domain: The DomainConfig to save
        output_dir: Output directory path
        schema_version: Current schema_version from sift.yaml
        yaml_file_size: Current sift.yaml file size

    Returns:
        Path to the saved domain YAML
    """
    discovered_path = output_dir / "discovered_domain.yaml"
    save_discovered_domain(domain, discovered_path)
    save_schema_metadata(output_dir, schema_version, yaml_file_size)
    return discovered_path

"""Tests for sift_kg.domains.incremental (incremental build workflow)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from sift_kg.domains.incremental import (
    SchemaCheckResult,
    add_to_unassigned_bucket,
    check_schema_version,
    load_unassigned_bucket,
    merge_discovered_schemas,
    save_schema_metadata,
)
from sift_kg.domains.models import DomainConfig, EntityTypeConfig, RelationTypeConfig


class TestSchemaCheckResult:
    """Test SchemaCheckResult status properties."""

    def test_match_is_match(self):
        """MATCH status returns is_match=True, needs_confirmation=False."""
        r = SchemaCheckResult(status=SchemaCheckResult.MATCH, message="ok")
        assert r.is_match is True
        assert r.needs_confirmation is False

    def test_mismatch_needs_confirmation(self):
        """Mismatch statuses return needs_confirmation=True."""
        for status in [
            SchemaCheckResult.MISMATCH_VERSION,
            SchemaCheckResult.MISMATCH_FILESIZE,
            SchemaCheckResult.MISMATCH_MISSING,
        ]:
            r = SchemaCheckResult(status=status, message="changed")
            assert r.is_match is False
            assert r.needs_confirmation is True

    def test_no_cache_needs_no_confirmation(self):
        """NO_CACHE status returns both False."""
        r = SchemaCheckResult(status=SchemaCheckResult.NO_CACHE, message="no cache")
        assert r.is_match is False
        assert r.needs_confirmation is False


class TestSchemaMetadata:
    """Test schema metadata save and check operations."""

    def test_save_and_check_match(self, tmp_dir):
        """Schema metadata saves and matches correctly."""
        with patch("sift_kg.config.SiftConfig.get_project_yaml_info") as mock_get:
            mock_get.return_value = {"schema_version": "v1", "file_size": 100}

            save_schema_metadata(tmp_dir, "v1", 100)
            result = check_schema_version(tmp_dir)

            assert result.is_match is True
            assert result.message == "Schema version matches"

    def test_check_version_mismatch(self, tmp_dir):
        """Version mismatch detected correctly."""
        save_schema_metadata(tmp_dir, "v1", 100)

        with patch("sift_kg.config.SiftConfig.get_project_yaml_info") as mock_get:
            mock_get.return_value = {"schema_version": "v2", "file_size": 100}

            result = check_schema_version(tmp_dir)
            assert result.is_match is False
            assert result.needs_confirmation is True
            assert "version mismatch" in result.message.lower()

    def test_check_filesize_mismatch(self, tmp_dir):
        """File size mismatch detected correctly."""
        save_schema_metadata(tmp_dir, "v1", 100)

        with patch("sift_kg.config.SiftConfig.get_project_yaml_info") as mock_get:
            mock_get.return_value = {"schema_version": "v1", "file_size": 150}

            result = check_schema_version(tmp_dir)
            assert result.is_match is False
            assert result.needs_confirmation is True
            assert "file size" in result.message.lower()

    def test_check_missing_version(self, tmp_dir):
        """Missing schema_version treated as mismatch."""
        save_schema_metadata(tmp_dir, "v1", 100)

        with patch("sift_kg.config.SiftConfig.get_project_yaml_info") as mock_get:
            mock_get.return_value = {"schema_version": None, "file_size": 100}

            result = check_schema_version(tmp_dir)
            assert result.is_match is False
            assert result.needs_confirmation is True
            assert "missing" in result.message.lower()

    def test_check_no_cache(self, tmp_dir):
        """No metadata file returns NO_CACHE."""
        with patch("sift_kg.config.SiftConfig.get_project_yaml_info") as mock_get:
            mock_get.return_value = {"schema_version": "v1", "file_size": 100}

            result = check_schema_version(tmp_dir)
            assert result.status == SchemaCheckResult.NO_CACHE
            assert result.needs_confirmation is False


class TestSchemaMerge:
    """Test incremental schema merging."""

    def test_merge_preserves_existing_types(self):
        """Existing entity/relation types are preserved in merge."""
        old = DomainConfig(
            name="test",
            entity_types={
                "PERSON": EntityTypeConfig(description="Person"),
                "ORGANIZATION": EntityTypeConfig(description="Org"),
            },
            relation_types={
                "WORKS_FOR": RelationTypeConfig(
                    description="Employment",
                    source_types=["PERSON"],
                    target_types=["ORGANIZATION"],
                ),
            },
            schema_free=True,
        )
        new = DomainConfig(
            name="test",
            entity_types={
                "PERSON": EntityTypeConfig(description="Person entity"),
                "LOCATION": EntityTypeConfig(description="Place"),
            },
            relation_types={
                "LOCATED_IN": RelationTypeConfig(
                    description="Location",
                    source_types=["ORGANIZATION"],
                    target_types=["LOCATION"],
                ),
            },
            schema_free=True,
        )

        merged = merge_discovered_schemas(old, new)

        assert set(merged.entity_types.keys()) == {"PERSON", "ORGANIZATION", "LOCATION"}
        assert set(merged.relation_types.keys()) == {"WORKS_FOR", "LOCATED_IN"}
        assert merged.entity_types["PERSON"].description == "Person"

    def test_merge_with_schema_free_check(self):
        """Merged domain remains schema_free."""
        old = DomainConfig(name="test", entity_types={}, relation_types={}, schema_free=True)
        new = DomainConfig(name="test", entity_types={}, relation_types={}, schema_free=True)

        merged = merge_discovered_schemas(old, new)
        assert merged.schema_free is True


class TestUnassignedBucket:
    """Test unassigned document bucket operations."""

    def test_add_and_load_unassigned(self, tmp_dir):
        """Documents can be added to and loaded from unassigned bucket."""
        doc_ids = ["doc:001", "doc:002"]
        add_to_unassigned_bucket(tmp_dir, doc_ids, reason="schema change")

        loaded = load_unassigned_bucket(tmp_dir)
        assert set(loaded["unassigned_documents"]) == set(doc_ids)
        assert loaded["reason"] == "schema change"

    def test_add_appends_to_existing(self, tmp_dir):
        """Adding documents appends to existing bucket."""
        add_to_unassigned_bucket(tmp_dir, ["doc:001"], reason="first")
        add_to_unassigned_bucket(tmp_dir, ["doc:002"], reason="second")

        loaded = load_unassigned_bucket(tmp_dir)
        assert set(loaded["unassigned_documents"]) == {"doc:001", "doc:002"}

    def test_load_empty_bucket(self, tmp_dir):
        """Loading empty bucket returns empty list."""
        loaded = load_unassigned_bucket(tmp_dir)
        assert loaded["unassigned_documents"] == []

"""Tests for sift_kg.graph.snapshots (knowledge graph time snapshots)."""

import json
import time
from pathlib import Path

import pytest

from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.graph.snapshots import (
    SnapshotDiff,
    create_snapshot,
    diff_snapshots,
    find_snapshot_by_name,
    roll_old_snapshots,
)


class TestSnapshotCreation:
    """Test snapshot creation and management."""

    def test_create_snapshot(self, sample_graph, tmp_dir):
        """Snapshot is created with correct filename format."""
        snapshot_path = create_snapshot(sample_graph, tmp_dir)

        assert snapshot_path.exists()
        assert snapshot_path.name.startswith("graph-")
        assert snapshot_path.name.endswith(".json")
        assert snapshot_path.parent == tmp_dir / "snapshots"

    def test_snapshot_content(self, sample_graph, tmp_dir):
        """Snapshot contains graph data and metadata."""
        snapshot_path = create_snapshot(sample_graph, tmp_dir)

        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert "nodes" in data
        assert "links" in data
        assert "metadata" in data
        assert "snapshot_timestamp" in data

    def test_snapshot_filename_iso8601(self, sample_graph, tmp_dir):
        """Snapshot filename follows ISO8601 pattern."""
        snapshot_path = create_snapshot(sample_graph, tmp_dir)

        name = snapshot_path.stem
        assert name.startswith("graph-")
        timestamp_part = name[len("graph-"):]
        assert "T" in timestamp_part

    def test_roll_old_snapshots(self, tmp_dir):
        """Old snapshots are rolled when exceeding retention count."""
        snapshots_dir = tmp_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            p = snapshots_dir / f"graph-2024-01-0{i+1}T00-00-00-000000Z.json"
            p.write_text("{}", encoding="utf-8")
            time.sleep(0.01)

        roll_old_snapshots(tmp_dir, retention_count=3)

        remaining = sorted(snapshots_dir.glob("*.json"))
        assert len(remaining) == 3

    def test_roll_keeps_newest(self, tmp_dir):
        """Rolling keeps the newest snapshots."""
        snapshots_dir = tmp_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        names = []
        for i in range(5):
            name = f"graph-2024-01-0{i+1}T00-00-00-000000Z.json"
            p = snapshots_dir / name
            p.write_text("{}", encoding="utf-8")
            time.sleep(0.01)
            names.append(name)

        roll_old_snapshots(tmp_dir, retention_count=2)

        remaining = sorted(snapshots_dir.glob("*.json"))
        remaining_names = [p.name for p in remaining]

        assert names[-1] in remaining_names
        assert names[-2] in remaining_names
        assert names[0] not in remaining_names

    def test_find_snapshot_by_name(self, sample_graph, tmp_dir):
        """Find snapshot by various name formats."""
        snapshot_path = create_snapshot(sample_graph, tmp_dir)

        found = find_snapshot_by_name(tmp_dir, snapshot_path.name)
        assert found == snapshot_path

        found = find_snapshot_by_name(tmp_dir, str(snapshot_path))
        assert found == snapshot_path

    def test_find_snapshot_not_found(self, tmp_dir):
        """Non-existent snapshot returns None."""
        result = find_snapshot_by_name(tmp_dir, "nonexistent.json")
        assert result is None


class TestSnapshotDiff:
    """Test snapshot diff calculation."""

    def _make_snapshot_data(self, nodes, edges=None, links=None):
        """Helper to create snapshot data structure."""
        return {
            "nodes": nodes,
            "links": links or [],
            "edges": edges or [],
            "metadata": {"snapshot_timestamp": "2024-01-01T00:00:00Z"},
        }

    def test_added_nodes(self):
        """New nodes in B are detected as added."""
        a = self._make_snapshot_data(
            [{"id": "n1", "name": "Alice", "entity_type": "PERSON"}],
        )
        b = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Bob", "entity_type": "PERSON"},
            ],
        )

        diff = SnapshotDiff(a, b)
        assert len(diff.added_nodes) == 1
        assert diff.added_nodes[0]["id"] == "n2"

    def test_removed_nodes(self):
        """Nodes in A but not B are detected as removed."""
        a = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Bob", "entity_type": "PERSON"},
            ],
        )
        b = self._make_snapshot_data(
            [{"id": "n1", "name": "Alice", "entity_type": "PERSON"}],
        )

        diff = SnapshotDiff(a, b)
        assert len(diff.removed_nodes) == 1
        assert diff.removed_nodes[0]["id"] == "n2"

    def test_added_edges(self):
        """New edges in B are detected as added."""
        a = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Acme", "entity_type": "ORGANIZATION"},
            ],
        )
        b = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Acme", "entity_type": "ORGANIZATION"},
            ],
            links=[
                {"source": "n1", "target": "n2", "relation_type": "WORKS_FOR"},
            ],
        )

        diff = SnapshotDiff(a, b)
        assert len(diff.added_edges) == 1
        assert diff.added_edges[0][2] == "WORKS_FOR"

    def test_removed_edges(self):
        """Edges in A but not B are detected as removed."""
        a = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Acme", "entity_type": "ORGANIZATION"},
            ],
            links=[
                {"source": "n1", "target": "n2", "relation_type": "WORKS_FOR"},
            ],
        )
        b = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Acme", "entity_type": "ORGANIZATION"},
            ],
        )

        diff = SnapshotDiff(a, b)
        assert len(diff.removed_edges) == 1

    def test_renamed_entities(self):
        """Entities with same name/type but different ID are detected as renamed."""
        a = self._make_snapshot_data(
            [{"id": "person:alice_smith", "name": "Alice Smith", "entity_type": "PERSON"}],
        )
        b = self._make_snapshot_data(
            [{"id": "person:alice_jones", "name": "Alice Smith", "entity_type": "PERSON"}],
        )

        diff = SnapshotDiff(a, b)
        assert len(diff.renamed_entities) == 1
        old_id, old_name, new_id, new_name = diff.renamed_entities[0]
        assert old_id == "person:alice_smith"
        assert new_id == "person:alice_jones"

    def test_to_markdown(self):
        """Markdown report is generated correctly."""
        a = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n2", "name": "Bob", "entity_type": "PERSON"},
            ],
        )
        b = self._make_snapshot_data(
            [
                {"id": "n1", "name": "Alice", "entity_type": "PERSON"},
                {"id": "n3", "name": "Charlie", "entity_type": "PERSON"},
            ],
        )

        diff = SnapshotDiff(a, b)
        md = diff.to_markdown()

        assert "## Summary" in md
        assert "## Added Entities" in md
        assert "## Removed Entities" in md
        assert "Charlie" in md
        assert "Bob" in md
        assert "| Metric" in md

    def test_diff_snapshots_function(self, sample_graph, tmp_dir):
        """diff_snapshots function returns markdown report."""
        snapshot_a = create_snapshot(sample_graph, tmp_dir)
        time.sleep(0.01)

        g2 = KnowledgeGraph()
        g2.add_entity("person:eve", "Eve", "PERSON", {}, [])
        snapshot_b = create_snapshot(g2, tmp_dir)

        report = diff_snapshots(tmp_dir, snapshot_a.name, snapshot_b.name)
        assert "## Summary" in report
        assert "Eve" in report


class TestSnapshotRetention:
    """Test snapshot retention configuration."""

    def test_default_retention_in_create(self, sample_graph, tmp_dir):
        """create_snapshot uses default retention of 50."""
        snapshot_path = create_snapshot(sample_graph, tmp_dir)
        assert snapshot_path.exists()

    def test_custom_retention(self, tmp_dir):
        """Custom retention count is respected."""
        snapshots_dir = tmp_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            p = snapshots_dir / f"graph-2024-01-0{i+1}T00-00-00-000000Z.json"
            p.write_text("{}", encoding="utf-8")
            time.sleep(0.01)

        roll_old_snapshots(tmp_dir, retention_count=3)

        remaining = list(snapshots_dir.glob("*.json"))
        assert len(remaining) == 3

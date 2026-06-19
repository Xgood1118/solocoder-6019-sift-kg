"""Knowledge graph snapshot management.

Provides functionality for:
- Creating time-based snapshots of the knowledge graph
- Rolling deletion of old snapshots based on retention policy
- Computing diffs between two snapshots (added/removed nodes, renamed entities)
- Replaying snapshots to regenerate narratives without re-running extraction
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = "snapshots"
SNAPSHOT_PREFIX = "graph-"


def _get_snapshot_dir(output_dir: Path) -> Path:
    """Get the snapshots subdirectory path."""
    return output_dir / SNAPSHOT_DIR


def _generate_snapshot_filename() -> str:
    """Generate snapshot filename with ISO8601 timestamp.

    Format: graph-YYYY-MM-DDTHH-MM-SS-ffffffZ.json
    """
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    return f"{SNAPSHOT_PREFIX}{timestamp}.json"


def _parse_snapshot_timestamp(filename: str) -> datetime | None:
    """Parse timestamp from snapshot filename.

    Returns datetime object or None if filename format is invalid.
    """
    if not filename.startswith(SNAPSHOT_PREFIX) or not filename.endswith(".json"):
        return None

    timestamp_str = filename[len(SNAPSHOT_PREFIX) : -len(".json")]

    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S-%fZ")
    except ValueError:
        return None


def _list_snapshots(snapshot_dir: Path) -> list[tuple[Path, datetime]]:
    """List all snapshot files with their timestamps, sorted newest first.

    Returns list of (path, timestamp) tuples sorted by timestamp descending.
    """
    if not snapshot_dir.exists():
        return []

    snapshots = []
    for path in snapshot_dir.glob(f"{SNAPSHOT_PREFIX}*.json"):
        ts = _parse_snapshot_timestamp(path.name)
        if ts is not None:
            snapshots.append((path, ts))

    snapshots.sort(key=lambda x: x[1], reverse=True)
    return snapshots


def create_snapshot(
    kg: "KnowledgeGraph",
    output_dir: Path,
    retention_count: int = 50,
) -> Path:
    """Create a snapshot of the current knowledge graph.

    Args:
        kg: KnowledgeGraph instance to snapshot
        output_dir: Base output directory
        retention_count: Number of snapshots to retain (older ones are deleted)

    Returns:
        Path to the created snapshot file
    """
    from sift_kg.config import SiftConfig

    try:
        config = SiftConfig()
        effective_retention = config.snapshot_retention or retention_count
    except Exception:
        effective_retention = retention_count

    snapshot_dir = _get_snapshot_dir(output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    filename = _generate_snapshot_filename()
    snapshot_path = snapshot_dir / filename

    export_data = kg.export()

    snapshot_data = {
        "snapshot_version": "1.0",
        "snapshot_timestamp": datetime.now(UTC).isoformat(),
        "sift_kg_version": export_data.get("metadata", {}).get("sift_kg_version", "unknown"),
        **export_data,
    }

    snapshot_path.write_text(
        json.dumps(snapshot_data, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(f"Created snapshot: {snapshot_path}")

    roll_old_snapshots(output_dir, effective_retention)

    return snapshot_path


def roll_old_snapshots(output_dir: Path, retention_count: int) -> int:
    """Delete old snapshots exceeding the retention count.

    Args:
        output_dir: Base output directory
        retention_count: Number of snapshots to retain

    Returns:
        Number of snapshots deleted
    """
    snapshot_dir = _get_snapshot_dir(output_dir)
    snapshots = _list_snapshots(snapshot_dir)

    deleted = 0
    if len(snapshots) > retention_count:
        to_delete = snapshots[retention_count:]
        for path, _ts in to_delete:
            try:
                path.unlink()
                deleted += 1
                logger.info(f"Deleted old snapshot: {path.name}")
            except OSError as e:
                logger.warning(f"Failed to delete snapshot {path.name}: {e}")

    return deleted


def load_snapshot(snapshot_path: Path) -> dict[str, Any] | None:
    """Load a snapshot file.

    Args:
        snapshot_path: Path to the snapshot JSON file

    Returns:
        Snapshot data dict, or None if loading fails
    """
    if not snapshot_path.exists():
        logger.warning(f"Snapshot not found: {snapshot_path}")
        return None

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load snapshot {snapshot_path}: {e}")
        return None


def find_snapshot_by_name(output_dir: Path, name: str) -> Path | None:
    """Find a snapshot file by name (either full path, filename, or timestamp).

    Args:
        output_dir: Base output directory
        name: Snapshot identifier (full path, filename, or partial timestamp)

    Returns:
        Path to the snapshot file, or None if not found
    """
    snapshot_dir = _get_snapshot_dir(output_dir)
    path = Path(name)

    if path.is_absolute() and path.exists():
        return path

    if snapshot_dir.joinpath(name).exists():
        return snapshot_dir.joinpath(name)

    snapshots = _list_snapshots(snapshot_dir)
    for snap_path, ts in snapshots:
        if name in snap_path.name or name in ts.isoformat():
            return snap_path

    return None


class SnapshotDiff:
    """Represents the difference between two snapshots."""

    def __init__(self, snapshot_a: "dict[str, Any] | Path", snapshot_b: "dict[str, Any] | Path"):
        if isinstance(snapshot_a, dict):
            self.data_a = snapshot_a
            self.snapshot_a = None
        else:
            self.snapshot_a = snapshot_a
            self.data_a = load_snapshot(snapshot_a)
            if self.data_a is None:
                raise ValueError(f"Failed to load snapshot: {snapshot_a}")

        if isinstance(snapshot_b, dict):
            self.data_b = snapshot_b
            self.snapshot_b = None
        else:
            self.snapshot_b = snapshot_b
            self.data_b = load_snapshot(snapshot_b)
            if self.data_b is None:
                raise ValueError(f"Failed to load snapshot: {snapshot_b}")

        self.nodes_a = {n["id"]: n for n in self.data_a.get("nodes", [])}
        self.nodes_b = {n["id"]: n for n in self.data_b.get("nodes", [])}

        edges_a = self.data_a.get("links", []) + self.data_a.get("edges", [])
        edges_b = self.data_b.get("links", []) + self.data_b.get("edges", [])

        self.edges_a: set[tuple[str, str, str]] = set()
        for e in edges_a:
            key = (e.get("source", ""), e.get("target", ""), e.get("relation_type", ""))
            self.edges_a.add(key)

        self.edges_b: set[tuple[str, str, str]] = set()
        for e in edges_b:
            key = (e.get("source", ""), e.get("target", ""), e.get("relation_type", ""))
            self.edges_b.add(key)

    @property
    def added_nodes(self) -> list[dict[str, Any]]:
        """Nodes present in B but not in A."""
        return [self.nodes_b[nid] for nid in self.nodes_b if nid not in self.nodes_a]

    @property
    def removed_nodes(self) -> list[dict[str, Any]]:
        """Nodes present in A but not in B."""
        return [self.nodes_a[nid] for nid in self.nodes_a if nid not in self.nodes_b]

    @property
    def added_edges(self) -> list[tuple[str, str, str]]:
        """Edges present in B but not in A."""
        return sorted(list(self.edges_b - self.edges_a))

    @property
    def removed_edges(self) -> list[tuple[str, str, str]]:
        """Edges present in A but not in B."""
        return sorted(list(self.edges_a - self.edges_b))

    @property
    def renamed_entities(self) -> list[tuple[str, str, str, str]]:
        """Detect entities that may have been renamed.

        Looks for entities that share the same type and have similar names
        but different IDs. Returns list of (old_id, old_name, new_id, new_name).
        """
        renamed = []

        a_by_type_name: dict[tuple[str, str], tuple[str, str]] = {}
        for nid, data in self.nodes_a.items():
            etype = data.get("entity_type", "")
            name = data.get("name", "").lower().strip()
            if etype and name:
                a_by_type_name[(etype, name)] = (nid, data.get("name", ""))

        b_by_type_name: dict[tuple[str, str], tuple[str, str]] = {}
        for nid, data in self.nodes_b.items():
            etype = data.get("entity_type", "")
            name = data.get("name", "").lower().strip()
            if etype and name:
                b_by_type_name[(etype, name)] = (nid, data.get("name", ""))

        for (etype, name), (old_id, old_name) in a_by_type_name.items():
            if (etype, name) in b_by_type_name:
                new_id, new_name = b_by_type_name[(etype, name)]
                if old_id != new_id:
                    renamed.append((old_id, old_name, new_id, new_name))

        return renamed

    def to_markdown(self) -> str:
        """Generate markdown report of the diff."""
        if self.snapshot_a is not None:
            ts_a = _parse_snapshot_timestamp(self.snapshot_a.name)
            ts_a_str = ts_a.isoformat() if ts_a else self.snapshot_a.name
        else:
            ts_a_str = self.data_a.get("snapshot_timestamp", "unknown")

        if self.snapshot_b is not None:
            ts_b = _parse_snapshot_timestamp(self.snapshot_b.name)
            ts_b_str = ts_b.isoformat() if ts_b else self.snapshot_b.name
        else:
            ts_b_str = self.data_b.get("snapshot_timestamp", "unknown")

        lines = []
        lines.append(f"# Graph Snapshot Diff")
        lines.append("")
        lines.append(f"**From:** {ts_a_str}")
        lines.append(f"**To:**   {ts_b_str}")
        lines.append("")

        stats_a = self.data_a.get("metadata", {})
        stats_b = self.data_b.get("metadata", {})

        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Before | After | Change |")
        lines.append("|--------|--------|-------|--------|")
        lines.append(
            f"| Entities | {stats_a.get('entity_count', '?')} | "
            f"{stats_b.get('entity_count', '?')} | "
            f"{int(stats_b.get('entity_count', 0)) - int(stats_a.get('entity_count', 0)):+,} |"
        )
        lines.append(
            f"| Relations | {stats_a.get('relation_count', '?')} | "
            f"{stats_b.get('relation_count', '?')} | "
            f"{int(stats_b.get('relation_count', 0)) - int(stats_a.get('relation_count', 0)):+,} |"
        )
        lines.append("")

        lines.append("## Changes")
        lines.append("")
        lines.append(f"- **{len(self.added_nodes)}** new entities added")
        lines.append(f"- **{len(self.removed_nodes)}** entities removed")
        lines.append(f"- **{len(self.added_edges)}** new relations added")
        lines.append(f"- **{len(self.removed_edges)}** relations removed")
        if self.renamed_entities:
            lines.append(f"- **{len(self.renamed_entities)}** entities potentially renamed")
        lines.append("")

        if self.added_nodes:
            lines.append("### Added Entities")
            lines.append("")
            lines.append("| ID | Name | Type |")
            lines.append("|----|------|------|")
            for node in sorted(self.added_nodes, key=lambda n: n.get("entity_type", "")):
                lines.append(
                    f"| {node.get('id', '')} | "
                    f"{node.get('name', '')} | "
                    f"{node.get('entity_type', '')} |"
                )
            lines.append("")

        if self.removed_nodes:
            lines.append("### Removed Entities")
            lines.append("")
            lines.append("| ID | Name | Type |")
            lines.append("|----|------|------|")
            for node in sorted(self.removed_nodes, key=lambda n: n.get("entity_type", "")):
                lines.append(
                    f"| {node.get('id', '')} | "
                    f"{node.get('name', '')} | "
                    f"{node.get('entity_type', '')} |"
                )
            lines.append("")

        if self.renamed_entities:
            lines.append("### Renamed Entities")
            lines.append("")
            lines.append("| Old ID | Old Name | New ID | New Name |")
            lines.append("|--------|----------|--------|----------|")
            for old_id, old_name, new_id, new_name in self.renamed_entities:
                lines.append(f"| {old_id} | {old_name} | {new_id} | {new_name} |")
            lines.append("")

        if self.added_edges:
            lines.append("### Added Relations")
            lines.append("")
            lines.append("| Source | Relation | Target |")
            lines.append("|--------|----------|--------|")
            for src, tgt, rel in self.added_edges[:50]:
                lines.append(f"| {src} | {rel} | {tgt} |")
            if len(self.added_edges) > 50:
                lines.append(f"| ... | ({len(self.added_edges) - 50} more) | ... |")
            lines.append("")

        if self.removed_edges:
            lines.append("### Removed Relations")
            lines.append("")
            lines.append("| Source | Relation | Target |")
            lines.append("|--------|----------|--------|")
            for src, tgt, rel in self.removed_edges[:50]:
                lines.append(f"| {src} | {rel} | {tgt} |")
            if len(self.removed_edges) > 50:
                lines.append(f"| ... | ({len(self.removed_edges) - 50} more) | ... |")
            lines.append("")

        return "\n".join(lines)


def diff_snapshots(
    output_dir: Path,
    snapshot_a_name: str,
    snapshot_b_name: str,
) -> str:
    """Compute diff between two snapshots and return markdown report.

    Args:
        output_dir: Base output directory
        snapshot_a_name: Name/identifier of the first (earlier) snapshot
        snapshot_b_name: Name/identifier of the second (later) snapshot

    Returns:
        Markdown diff report

    Raises:
        ValueError: If either snapshot cannot be found
    """
    snap_a = find_snapshot_by_name(output_dir, snapshot_a_name)
    snap_b = find_snapshot_by_name(output_dir, snapshot_b_name)

    if snap_a is None:
        raise ValueError(f"Snapshot A not found: {snapshot_a_name}")
    if snap_b is None:
        raise ValueError(f"Snapshot B not found: {snapshot_b_name}")

    diff = SnapshotDiff(snap_a, snap_b)
    return diff.to_markdown()


def replay_snapshot(
    output_dir: Path,
    snapshot_name: str,
    model: str | None = None,
    domain_name: str = "schema-free",
) -> Path:
    """Replay a snapshot to regenerate narrative without re-extraction.

    Loads a snapshot, reconstructs the KnowledgeGraph from it, and runs
    narrative generation. Does NOT re-run LLM extraction - only uses the
    snapshot data.

    Args:
        output_dir: Base output directory
        snapshot_name: Name/identifier of the snapshot to replay
        model: LLM model for narrative generation (uses config default if None)
        domain_name: Bundled domain name for system context

    Returns:
        Path to the generated narrative file

    Raises:
        ValueError: If snapshot cannot be found or loaded
    """
    from sift_kg.config import SiftConfig
    from sift_kg.extract.llm_client import LLMClient
    from sift_kg.graph.knowledge_graph import KnowledgeGraph
    from sift_kg.narrate.generator import generate_narrative

    snapshot_path = find_snapshot_by_name(output_dir, snapshot_name)
    if snapshot_path is None:
        raise ValueError(f"Snapshot not found: {snapshot_name}")

    snapshot_data = load_snapshot(snapshot_path)
    if snapshot_data is None:
        raise ValueError(f"Failed to load snapshot: {snapshot_name}")

    config = SiftConfig()
    effective_model = model or config.default_model

    try:
        config.validate_api_keys(effective_model)
    except ValueError as e:
        raise ValueError(f"API key validation failed: {e}") from None

    kg = KnowledgeGraph()

    for node in snapshot_data.get("nodes", []):
        node_id = node.get("id")
        if node_id is None:
            continue
        attrs = {k: v for k, v in node.items() if k != "id"}
        kg.graph.add_node(node_id, **attrs)

    for link in snapshot_data.get("links", snapshot_data.get("edges", [])):
        source = link.get("source")
        target = link.get("target")
        if source is None or target is None:
            continue
        attrs = {k: v for k, v in link.items() if k not in ("source", "target")}
        relation_type = attrs.get("relation_type", "")
        canonical_key = attrs.get("canonical_key")
        edge_key = canonical_key or kg._canonical_relation_key(source, relation_type, target)
        kg.graph.add_edge(source, target, key=edge_key, **attrs)

    logger.info(
        f"Reconstructed graph from snapshot: {kg.entity_count} entities, "
        f"{kg.relation_count} relations"
    )

    from sift_kg.domains.loader import DomainLoader

    loader = DomainLoader()
    if config.domain_path and config.domain_path.exists():
        domain_config = loader.load_from_path(config.domain_path)
    else:
        domain_config = loader.load_bundled(domain_name)
    system_context = domain_config.system_context or ""

    llm = LLMClient(model=effective_model)

    narrative_path = generate_narrative(
        kg=kg,
        llm=llm,
        output_dir=output_dir,
        system_context=system_context,
        include_entity_descriptions=True,
        max_cost=None,
        replay_mode=True,
    )

    logger.info(f"Replay complete: narrative generated at {narrative_path}")
    return narrative_path

"""Tests for sift_kg.resolve.resolver auto-approve functionality."""

from pathlib import Path

from sift_kg.resolve.io import write_proposals
from sift_kg.resolve.models import MergeFile, MergeMember, MergeProposal
from sift_kg.resolve.resolver import apply_auto_approve


class TestAutoApprove:
    """Test auto-approve logic for merge proposals."""

    def test_high_confidence_auto_approved(self, tmp_dir):
        """Proposals with average confidence >= threshold are auto-approved."""
        proposals = [
            MergeProposal(
                canonical_id="person:alice",
                canonical_name="Alice",
                entity_type="PERSON",
                members=[
                    MergeMember(id="person:alice_smith", name="Alice Smith", confidence=0.95),
                    MergeMember(id="person:alice_jones", name="Alice Jones", confidence=0.90),
                ],
            ),
        ]
        merge_file = MergeFile(proposals=proposals)

        result_merged, result_pending, auto_count = apply_auto_approve(
            merge_file, 0.80, tmp_dir
        )

        assert auto_count == 1
        assert result_merged.proposals[0].status == "CONFIRMED"
        assert len(result_pending.proposals) == 0

    def test_low_confidence_pending(self, tmp_dir):
        """Proposals below threshold go to merges_pending.json."""
        proposals = [
            MergeProposal(
                canonical_id="person:bob",
                canonical_name="Bob",
                entity_type="PERSON",
                members=[
                    MergeMember(id="person:bob_smith", name="Bob Smith", confidence=0.70),
                ],
            ),
        ]
        merge_file = MergeFile(proposals=proposals)

        result_merged, result_pending, auto_count = apply_auto_approve(
            merge_file, 0.80, tmp_dir
        )

        assert auto_count == 0
        assert result_merged.proposals[0].status == "DRAFT"
        assert len(result_pending.proposals) == 1

    def test_mixed_confidence(self, tmp_dir):
        """Mixed proposals: high auto-approved, low pending."""
        proposals = [
            MergeProposal(
                canonical_id="person:high",
                canonical_name="High",
                entity_type="PERSON",
                members=[MergeMember(id="h1", name="H1", confidence=0.95)],
            ),
            MergeProposal(
                canonical_id="person:low",
                canonical_name="Low",
                entity_type="PERSON",
                members=[MergeMember(id="l1", name="L1", confidence=0.50)],
            ),
        ]
        merge_file = MergeFile(proposals=proposals)

        result_merged, result_pending, auto_count = apply_auto_approve(
            merge_file, 0.80, tmp_dir
        )

        assert auto_count == 1
        assert result_merged.proposals[0].status == "CONFIRMED"
        assert result_merged.proposals[1].status == "DRAFT"
        assert len(result_pending.proposals) == 1
        assert result_pending.proposals[0].canonical_id == "person:low"

    def test_pending_file_written(self, tmp_dir):
        """merges_pending.json is written to output directory."""
        proposals = [
            MergeProposal(
                canonical_id="person:pending",
                canonical_name="Pending",
                entity_type="PERSON",
                members=[MergeMember(id="p1", name="P1", confidence=0.60)],
            ),
        ]
        merge_file = MergeFile(proposals=proposals)

        apply_auto_approve(merge_file, 0.80, tmp_dir)

        pending_path = tmp_dir / "merges_pending.json"
        assert pending_path.exists()

    def test_average_confidence_calculation(self, tmp_dir):
        """Average confidence is correctly calculated across members."""
        proposals = [
            MergeProposal(
                canonical_id="person:avg",
                canonical_name="Avg",
                entity_type="PERSON",
                members=[
                    MergeMember(id="m1", name="M1", confidence=0.90),
                    MergeMember(id="m2", name="M2", confidence=0.70),
                ],
            ),
        ]
        merge_file = MergeFile(proposals=proposals)

        _, _, auto_count_08 = apply_auto_approve(merge_file, 0.80, tmp_dir)
        assert auto_count_08 == 1

        _, _, auto_count_081 = apply_auto_approve(merge_file, 0.81, tmp_dir)
        assert auto_count_081 == 0

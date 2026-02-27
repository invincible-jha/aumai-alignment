"""Shared test fixtures for aumai-alignment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aumai_alignment.core import DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset, MarketplaceListing


@pytest.fixture()
def sample_dataset() -> AlignmentDataset:
    """Minimal valid AlignmentDataset for testing."""
    return AlignmentDataset(
        dataset_id="ds-001",
        name="Safety Prompts v1",
        description="A dataset of safety-related prompts for alignment evaluation.",
        category="safety",
        size=500,
        format="jsonl",
        license="CC-BY-4.0",
        tags=["safety", "harmlessness", "prompts"],
        quality_score=0.85,
    )


@pytest.fixture()
def high_quality_dataset() -> AlignmentDataset:
    """AlignmentDataset with maximum quality score."""
    return AlignmentDataset(
        dataset_id="ds-002",
        name="Helpfulness Eval",
        description="Evaluations for helpfulness across diverse tasks.",
        category="helpfulness",
        size=1000,
        format="json",
        license="MIT",
        tags=["helpfulness", "eval"],
        quality_score=0.95,
    )


@pytest.fixture()
def low_quality_dataset() -> AlignmentDataset:
    """AlignmentDataset with low quality score."""
    return AlignmentDataset(
        dataset_id="ds-003",
        name="Rough Drafts",
        description="Unverified community-contributed examples.",
        category="safety",
        size=200,
        format="csv",
        license="CC0",
        tags=["community"],
        quality_score=0.30,
    )


@pytest.fixture()
def registry(
    sample_dataset: AlignmentDataset,
    high_quality_dataset: AlignmentDataset,
) -> DatasetRegistry:
    """Pre-populated DatasetRegistry with two datasets."""
    reg = DatasetRegistry()
    reg.register(sample_dataset)
    reg.register(high_quality_dataset)
    return reg


@pytest.fixture()
def runner(registry: DatasetRegistry) -> EvaluationRunner:
    """EvaluationRunner backed by the pre-populated registry."""
    return EvaluationRunner(registry=registry)


@pytest.fixture()
def dataset_json_file(sample_dataset: AlignmentDataset, tmp_path: Path) -> Path:
    """Write a valid dataset JSON file to tmp_path and return the path."""
    data = sample_dataset.model_dump(mode="json")
    file_path = tmp_path / "dataset.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


@pytest.fixture()
def dataset_yaml_file(sample_dataset: AlignmentDataset, tmp_path: Path) -> Path:
    """Write a valid dataset YAML file to tmp_path and return the path."""
    import yaml  # type: ignore[import-untyped]

    data = sample_dataset.model_dump(mode="json")
    file_path = tmp_path / "dataset.yaml"
    file_path.write_text(yaml.dump(data), encoding="utf-8")
    return file_path

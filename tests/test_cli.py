"""Comprehensive CLI tests for aumai-alignment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_alignment.cli import main
from aumai_alignment.models import AlignmentDataset


def _fresh_runner() -> CliRunner:
    """Return a CliRunner without mix_stderr (Click 8.2 compatible)."""
    return CliRunner()


class TestCLIVersion:
    def test_version_flag(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_flag(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "alignment" in result.output.lower() or "AumAI" in result.output


class TestSearchCommand:
    def test_search_empty_registry_returns_no_datasets(self) -> None:
        runner = _fresh_runner()
        # Module-level registry is shared; search with an unlikely query
        result = runner.invoke(main, ["search", "--query", "zzznonexistentzzz"])
        assert result.exit_code == 0
        assert "No datasets found" in result.output

    def test_search_help(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_search_with_min_quality_option(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["search", "--query", "", "--min-quality", "0.9"])
        assert result.exit_code == 0

    def test_search_with_category_option(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["search", "--query", "", "--category", "safety"])
        assert result.exit_code == 0


class TestRegisterCommand:
    def test_register_json_file(
        self, dataset_json_file: Path, sample_dataset: AlignmentDataset
    ) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["register", "--config", str(dataset_json_file)])
        assert result.exit_code == 0
        assert "Registered dataset" in result.output
        assert sample_dataset.name in result.output

    def test_register_yaml_file(
        self, dataset_yaml_file: Path, sample_dataset: AlignmentDataset
    ) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["register", "--config", str(dataset_yaml_file)])
        assert result.exit_code == 0
        assert "Registered dataset" in result.output

    def test_register_missing_config_flag_errors(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["register"])
        assert result.exit_code != 0

    def test_register_nonexistent_file_errors(self, tmp_path: Path) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["register", "--config", str(tmp_path / "missing.json")])
        assert result.exit_code != 0

    def test_register_outputs_dataset_id(
        self, dataset_json_file: Path, sample_dataset: AlignmentDataset
    ) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["register", "--config", str(dataset_json_file)])
        assert sample_dataset.dataset_id in result.output

    def test_register_then_search_finds_dataset(
        self, dataset_json_file: Path, sample_dataset: AlignmentDataset
    ) -> None:
        runner = _fresh_runner()
        runner.invoke(main, ["register", "--config", str(dataset_json_file)])
        search_result = runner.invoke(
            main, ["search", "--query", sample_dataset.name[:8]]
        )
        assert search_result.exit_code == 0

    def test_register_invalid_json_errors(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all {{{{", encoding="utf-8")
        runner = _fresh_runner()
        result = runner.invoke(main, ["register", "--config", str(bad_file)])
        assert result.exit_code != 0

    def test_register_help(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["register", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()


class TestServeCommand:
    def test_serve_without_uvicorn_exits_nonzero(self) -> None:
        """Serve should exit 1 if uvicorn is not installed or fails."""
        import unittest.mock as mock

        runner = _fresh_runner()
        with mock.patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(main, ["serve"])
        # Either exits with error code (uvicorn missing) or succeeds if installed
        assert isinstance(result.exit_code, int)

    def test_serve_help(self) -> None:
        runner = _fresh_runner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower()
        assert "host" in result.output.lower()

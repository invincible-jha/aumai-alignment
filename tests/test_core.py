"""Comprehensive tests for aumai-alignment core module."""

from __future__ import annotations

import pytest

from aumai_alignment.core import (
    DatasetNotFoundError,
    DatasetRegistry,
    EvaluationRunner,
    _default_scorer,
)
from aumai_alignment.models import AlignmentDataset, EvaluationResult, MarketplaceListing


# ---------------------------------------------------------------------------
# AlignmentDataset model tests
# ---------------------------------------------------------------------------


class TestAlignmentDatasetModel:
    def test_valid_dataset_creation(self, sample_dataset: AlignmentDataset) -> None:
        assert sample_dataset.dataset_id == "ds-001"
        assert sample_dataset.name == "Safety Prompts v1"
        assert sample_dataset.quality_score == 0.85

    def test_dataset_quality_score_bounds(self) -> None:
        with pytest.raises(Exception):
            AlignmentDataset(
                dataset_id="x",
                name="x",
                description="x",
                category="x",
                size=0,
                format="json",
                license="MIT",
                quality_score=1.1,  # exceeds upper bound
            )

    def test_dataset_quality_score_lower_bound(self) -> None:
        with pytest.raises(Exception):
            AlignmentDataset(
                dataset_id="x",
                name="x",
                description="x",
                category="x",
                size=0,
                format="json",
                license="MIT",
                quality_score=-0.1,  # below lower bound
            )

    def test_dataset_size_non_negative(self) -> None:
        with pytest.raises(Exception):
            AlignmentDataset(
                dataset_id="x",
                name="x",
                description="x",
                category="x",
                size=-1,  # negative size
                format="json",
                license="MIT",
                quality_score=0.5,
            )

    def test_dataset_tags_default_empty(self) -> None:
        ds = AlignmentDataset(
            dataset_id="x",
            name="x",
            description="x",
            category="x",
            size=10,
            format="json",
            license="MIT",
            quality_score=0.5,
        )
        assert ds.tags == []

    def test_dataset_download_url_optional(self, sample_dataset: AlignmentDataset) -> None:
        assert sample_dataset.download_url is None

    def test_dataset_with_download_url(self) -> None:
        ds = AlignmentDataset(
            dataset_id="x",
            name="x",
            description="x",
            category="x",
            size=10,
            format="json",
            license="MIT",
            quality_score=0.5,
            download_url="https://example.com/dataset.zip",
        )
        assert ds.download_url == "https://example.com/dataset.zip"


# ---------------------------------------------------------------------------
# MarketplaceListing model tests
# ---------------------------------------------------------------------------


class TestMarketplaceListingModel:
    def test_default_listing_values(self, sample_dataset: AlignmentDataset) -> None:
        listing = MarketplaceListing(dataset=sample_dataset)
        assert listing.downloads == 0
        assert listing.rating == 0.0
        assert listing.reviews == 0

    def test_rating_upper_bound(self, sample_dataset: AlignmentDataset) -> None:
        with pytest.raises(Exception):
            MarketplaceListing(dataset=sample_dataset, rating=5.1)

    def test_rating_lower_bound(self, sample_dataset: AlignmentDataset) -> None:
        with pytest.raises(Exception):
            MarketplaceListing(dataset=sample_dataset, rating=-0.1)

    def test_downloads_non_negative(self, sample_dataset: AlignmentDataset) -> None:
        with pytest.raises(Exception):
            MarketplaceListing(dataset=sample_dataset, downloads=-1)


# ---------------------------------------------------------------------------
# DatasetRegistry tests
# ---------------------------------------------------------------------------


class TestDatasetRegistry:
    def test_register_creates_listing(self, sample_dataset: AlignmentDataset) -> None:
        reg = DatasetRegistry()
        reg.register(sample_dataset)
        retrieved = reg.get(sample_dataset.dataset_id)
        assert retrieved.dataset_id == sample_dataset.dataset_id

    def test_get_returns_correct_dataset(self, registry: DatasetRegistry, sample_dataset: AlignmentDataset) -> None:
        result = registry.get("ds-001")
        assert result.name == sample_dataset.name

    def test_get_raises_on_missing(self, registry: DatasetRegistry) -> None:
        with pytest.raises(DatasetNotFoundError):
            registry.get("nonexistent-id")

    def test_dataset_not_found_error_is_key_error(self, registry: DatasetRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("nonexistent-id")

    def test_register_overwrites_dataset_preserves_downloads(
        self, registry: DatasetRegistry, sample_dataset: AlignmentDataset
    ) -> None:
        registry.increment_downloads("ds-001")
        # Re-register the same dataset — downloads should be preserved
        registry.register(sample_dataset)
        listing = registry._listings["ds-001"]
        assert listing.downloads == 1

    def test_register_new_listing_starts_zero_downloads(self, sample_dataset: AlignmentDataset) -> None:
        reg = DatasetRegistry()
        reg.register(sample_dataset)
        assert reg._listings["ds-001"].downloads == 0

    def test_increment_downloads_increases_count(self, registry: DatasetRegistry) -> None:
        registry.increment_downloads("ds-001")
        registry.increment_downloads("ds-001")
        assert registry._listings["ds-001"].downloads == 2

    def test_increment_downloads_no_op_for_missing(self, registry: DatasetRegistry) -> None:
        # Should not raise; just silently skip
        registry.increment_downloads("missing-id")

    def test_search_empty_query_returns_all(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="")
        assert len(results) == 2

    def test_search_by_name(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="Safety Prompts")
        assert len(results) == 1
        assert results[0].dataset.dataset_id == "ds-001"

    def test_search_by_description_keyword(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="diverse tasks")
        assert len(results) == 1
        assert results[0].dataset.dataset_id == "ds-002"

    def test_search_by_tag(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="harmlessness")
        assert len(results) == 1
        assert results[0].dataset.dataset_id == "ds-001"

    def test_search_case_insensitive(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="SAFETY PROMPTS")
        assert len(results) == 1

    def test_search_no_match_returns_empty(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="zzzzunknown")
        assert results == []

    def test_search_by_category(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="", category="safety")
        assert len(results) == 1
        assert results[0].dataset.dataset_id == "ds-001"

    def test_search_by_category_case_insensitive(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="", category="HELPFULNESS")
        assert len(results) == 1

    def test_search_category_no_match(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="", category="unknown-category")
        assert results == []

    def test_search_min_quality_filters_low(
        self,
        registry: DatasetRegistry,
        low_quality_dataset: AlignmentDataset,
    ) -> None:
        registry.register(low_quality_dataset)
        results = registry.search(query="", min_quality=0.50)
        ids = {r.dataset.dataset_id for r in results}
        assert "ds-003" not in ids

    def test_search_results_sorted_descending_quality(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="")
        scores = [r.dataset.quality_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_combined_query_and_category(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="safety", category="safety")
        assert len(results) == 1

    def test_search_combined_no_match(self, registry: DatasetRegistry) -> None:
        results = registry.search(query="helpfulness", category="safety")
        assert results == []


# ---------------------------------------------------------------------------
# EvaluationRunner tests
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    def test_evaluate_returns_result(
        self, runner: EvaluationRunner, sample_dataset: AlignmentDataset
    ) -> None:
        outputs = [{"score": 0.8}, {"score": 0.6}]
        result = runner.evaluate("ds-001", outputs, model_name="test-model")
        assert isinstance(result, EvaluationResult)
        assert result.dataset_id == "ds-001"
        assert result.model_name == "test-model"

    def test_evaluate_aggregate_score_is_average(self, runner: EvaluationRunner) -> None:
        outputs = [{"score": 0.8}, {"score": 0.6}]
        result = runner.evaluate("ds-001", outputs)
        assert result.score == pytest.approx(0.7, abs=1e-4)

    def test_evaluate_empty_outputs_returns_zero_score(self, runner: EvaluationRunner) -> None:
        result = runner.evaluate("ds-001", [], model_name="empty-model")
        assert result.score == 0.0
        assert result.metrics["sample_count"] == 0.0

    def test_evaluate_raises_for_missing_dataset(self, runner: EvaluationRunner) -> None:
        with pytest.raises(DatasetNotFoundError):
            runner.evaluate("nonexistent", [{"score": 0.5}])

    def test_evaluate_metrics_contain_expected_keys(self, runner: EvaluationRunner) -> None:
        outputs = [{"score": 0.9}, {"score": 0.3}]
        result = runner.evaluate("ds-001", outputs)
        assert "mean_score" in result.metrics
        assert "min_score" in result.metrics
        assert "max_score" in result.metrics
        assert "sample_count" in result.metrics

    def test_evaluate_metrics_min_max(self, runner: EvaluationRunner) -> None:
        outputs = [{"score": 0.9}, {"score": 0.3}, {"score": 0.6}]
        result = runner.evaluate("ds-001", outputs)
        assert result.metrics["min_score"] == pytest.approx(0.3)
        assert result.metrics["max_score"] == pytest.approx(0.9)

    def test_evaluate_score_clamped_to_unit_interval(self, runner: EvaluationRunner) -> None:
        # Default scorer clamps to [0, 1]
        outputs = [{"score": 2.0}, {"score": -1.0}]
        result = runner.evaluate("ds-001", outputs)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_stores_result(self, runner: EvaluationRunner) -> None:
        runner.evaluate("ds-001", [{"score": 0.5}])
        results = runner.get_results("ds-001")
        assert len(results) == 1

    def test_evaluate_accumulates_multiple_results(self, runner: EvaluationRunner) -> None:
        runner.evaluate("ds-001", [{"score": 0.5}])
        runner.evaluate("ds-001", [{"score": 0.8}])
        results = runner.get_results("ds-001")
        assert len(results) == 2

    def test_get_results_returns_empty_for_unknown(self, runner: EvaluationRunner) -> None:
        assert runner.get_results("never-evaluated") == []

    def test_evaluate_uses_custom_scorer(self, registry: DatasetRegistry) -> None:
        custom_scorer = lambda output: 1.0  # always returns 1.0  # noqa: E731
        custom_runner = EvaluationRunner(registry=registry, scoring_fn=custom_scorer)
        result = custom_runner.evaluate("ds-001", [{"text": "hello"}, {"text": "world"}])
        assert result.score == 1.0

    def test_evaluate_model_name_defaults_to_unknown(self, runner: EvaluationRunner) -> None:
        result = runner.evaluate("ds-001", [{"score": 0.5}])
        assert result.model_name == "unknown"

    def test_evaluate_score_rounded_to_4_decimals(self, runner: EvaluationRunner) -> None:
        outputs = [{"score": 0.123456789}]
        result = runner.evaluate("ds-001", outputs)
        # Score is rounded to 4 decimal places
        assert result.score == round(0.123456789, 4)


# ---------------------------------------------------------------------------
# _default_scorer tests
# ---------------------------------------------------------------------------


class TestDefaultScorer:
    def test_returns_score_value(self) -> None:
        assert _default_scorer({"score": 0.75}) == pytest.approx(0.75)

    def test_clamps_above_one(self) -> None:
        assert _default_scorer({"score": 5.0}) == 1.0

    def test_clamps_below_zero(self) -> None:
        assert _default_scorer({"score": -3.0}) == 0.0

    def test_defaults_to_half_when_no_score_key(self) -> None:
        assert _default_scorer({"text": "hello"}) == 0.5

    def test_non_numeric_score_returns_half(self) -> None:
        assert _default_scorer({"score": "high"}) == 0.5

    def test_integer_score_treated_as_float(self) -> None:
        assert _default_scorer({"score": 1}) == 1.0

    def test_boolean_true_treated_as_one(self) -> None:
        # bool is subclass of int in Python
        assert _default_scorer({"score": True}) == 1.0

    def test_zero_score(self) -> None:
        assert _default_scorer({"score": 0.0}) == 0.0

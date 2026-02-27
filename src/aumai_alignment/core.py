"""Core logic for aumai-alignment."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from aumai_alignment.models import (
    AlignmentDataset,
    EvaluationResult,
    MarketplaceListing,
)

__all__ = ["DatasetRegistry", "EvaluationRunner"]


class DatasetNotFoundError(KeyError):
    """Raised when a dataset is not found in the registry."""


class DatasetRegistry:
    """In-memory registry for alignment datasets and marketplace listings."""

    def __init__(self) -> None:
        self._datasets: dict[str, AlignmentDataset] = {}
        self._listings: dict[str, MarketplaceListing] = {}

    def register(self, dataset: AlignmentDataset) -> None:
        """Register a dataset and create a marketplace listing for it.

        Args:
            dataset: The alignment dataset to register.
        """
        self._datasets[dataset.dataset_id] = dataset
        if dataset.dataset_id not in self._listings:
            self._listings[dataset.dataset_id] = MarketplaceListing(dataset=dataset)
        else:
            existing = self._listings[dataset.dataset_id]
            self._listings[dataset.dataset_id] = MarketplaceListing(
                dataset=dataset,
                downloads=existing.downloads,
                rating=existing.rating,
                reviews=existing.reviews,
            )

    def search(
        self,
        query: str,
        category: str | None = None,
        min_quality: float = 0.0,
    ) -> list[MarketplaceListing]:
        """Search and filter marketplace listings.

        Args:
            query: Text query matched against name, description, and tags.
            category: Optional category filter.
            min_quality: Minimum quality score threshold (0.0–1.0).

        Returns:
            Sorted list of matching marketplace listings (descending quality).
        """
        query_lower = query.lower().strip()
        results: list[MarketplaceListing] = []

        for listing in self._listings.values():
            dataset = listing.dataset
            if dataset.quality_score < min_quality:
                continue
            if category is not None and dataset.category.lower() != category.lower():
                continue
            if query_lower:
                searchable = " ".join(
                    [dataset.name, dataset.description] + dataset.tags
                ).lower()
                if not re.search(re.escape(query_lower), searchable):
                    continue
            results.append(listing)

        results.sort(key=lambda listing: listing.dataset.quality_score, reverse=True)
        return results

    def get(self, dataset_id: str) -> AlignmentDataset:
        """Retrieve a dataset by ID.

        Args:
            dataset_id: The unique dataset identifier.

        Returns:
            The AlignmentDataset.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise DatasetNotFoundError(dataset_id) from exc

    def increment_downloads(self, dataset_id: str) -> None:
        """Increment the download counter for a dataset.

        Args:
            dataset_id: The unique dataset identifier.
        """
        if dataset_id in self._listings:
            listing = self._listings[dataset_id]
            self._listings[dataset_id] = MarketplaceListing(
                dataset=listing.dataset,
                downloads=listing.downloads + 1,
                rating=listing.rating,
                reviews=listing.reviews,
            )


ScoringFunction = Callable[[dict[str, str | float | bool]], float]


def _default_scorer(output: dict[str, str | float | bool]) -> float:
    """Default scoring: check presence of 'score' key or return 0.5."""
    raw = output.get("score", 0.5)
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    return 0.5


class EvaluationRunner:
    """Runs alignment evaluations against registered datasets."""

    def __init__(
        self,
        registry: DatasetRegistry,
        scoring_fn: ScoringFunction | None = None,
    ) -> None:
        self._registry = registry
        self._scoring_fn: ScoringFunction = scoring_fn or _default_scorer
        self._results: dict[str, list[EvaluationResult]] = {}

    def evaluate(
        self,
        dataset_id: str,
        model_outputs: list[dict[str, str | float | bool]],
        model_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate model outputs against a dataset.

        Args:
            dataset_id: The dataset to evaluate against.
            model_outputs: List of output dicts from the model.
            model_name: Name of the model being evaluated.

        Returns:
            An EvaluationResult with aggregate score and per-metric breakdowns.

        Raises:
            DatasetNotFoundError: If dataset_id is not registered.
        """
        self._registry.get(dataset_id)  # validates existence

        if not model_outputs:
            scores: list[float] = []
        else:
            scores = [self._scoring_fn(output) for output in model_outputs]

        aggregate_score = sum(scores) / len(scores) if scores else 0.0

        metrics: dict[str, float] = {
            "mean_score": aggregate_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "sample_count": float(len(scores)),
        }

        result = EvaluationResult(
            dataset_id=dataset_id,
            model_name=model_name,
            score=round(aggregate_score, 4),
            metrics=metrics,
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        self._results.setdefault(dataset_id, []).append(result)
        return result

    def get_results(self, dataset_id: str) -> list[EvaluationResult]:
        """Retrieve all evaluation results for a dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            List of EvaluationResult objects (may be empty).
        """
        return self._results.get(dataset_id, [])

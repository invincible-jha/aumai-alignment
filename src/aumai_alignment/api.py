"""FastAPI application for aumai-alignment."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from aumai_alignment.core import DatasetNotFoundError, DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset, EvaluationResult, MarketplaceListing

app = FastAPI(title="AumAI Alignment Marketplace", version="0.1.0")

_registry = DatasetRegistry()
_runner = EvaluationRunner(registry=_registry)


@app.get("/api/datasets", response_model=list[MarketplaceListing])
def list_datasets(
    query: str = "",
    category: str | None = None,
    min_quality: float = 0.0,
) -> list[MarketplaceListing]:
    """List and search alignment datasets."""
    return _registry.search(query=query, category=category, min_quality=min_quality)


@app.get("/api/datasets/{dataset_id}", response_model=AlignmentDataset)
def get_dataset(dataset_id: str) -> AlignmentDataset:
    """Retrieve a single dataset by ID."""
    try:
        return _registry.get(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found") from exc


@app.post("/api/datasets", response_model=AlignmentDataset, status_code=201)
def register_dataset(dataset: AlignmentDataset) -> AlignmentDataset:
    """Register a new alignment dataset."""
    _registry.register(dataset)
    return dataset


@app.get("/api/evaluations/{dataset_id}", response_model=list[EvaluationResult])
def get_evaluations(dataset_id: str) -> list[EvaluationResult]:
    """Get all evaluation results for a dataset."""
    try:
        _registry.get(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found") from exc
    return _runner.get_results(dataset_id)

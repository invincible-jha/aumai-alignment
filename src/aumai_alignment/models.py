"""Pydantic models for aumai-alignment."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

__all__ = [
    "AlignmentDataset",
    "EvaluationResult",
    "MarketplaceListing",
]


class AlignmentDataset(BaseModel):
    """Represents an alignment dataset in the marketplace."""

    dataset_id: str
    name: str
    description: str
    category: str
    size: int = Field(ge=0, description="Number of samples")
    format: str
    license: str
    tags: list[str] = Field(default_factory=list)
    download_url: str | None = None
    quality_score: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """Result of evaluating a model against an alignment dataset."""

    dataset_id: str
    model_name: str
    score: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class MarketplaceListing(BaseModel):
    """Marketplace listing wrapping a dataset with marketplace metadata."""

    dataset: AlignmentDataset
    downloads: int = Field(ge=0, default=0)
    rating: float = Field(ge=0.0, le=5.0, default=0.0)
    reviews: int = Field(ge=0, default=0)

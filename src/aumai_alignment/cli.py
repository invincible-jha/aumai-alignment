"""CLI entry point for aumai-alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from aumai_alignment.core import DatasetNotFoundError, DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset

_registry = DatasetRegistry()
_runner = EvaluationRunner(registry=_registry)


@click.group()
@click.version_option()
def main() -> None:
    """AumAI Alignment — AI alignment dataset marketplace CLI."""


@main.command("search")
@click.option("--query", default="", show_default=True, help="Search query string.")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--min-quality", default=0.0, show_default=True, type=float, help="Minimum quality score.")
def search(query: str, category: str | None, min_quality: float) -> None:
    """Search for alignment datasets in the registry."""
    results = _registry.search(query=query, category=category, min_quality=min_quality)
    if not results:
        click.echo("No datasets found matching your criteria.")
        return
    for listing in results:
        ds = listing.dataset
        click.echo(
            f"[{ds.dataset_id}] {ds.name}  quality={ds.quality_score:.2f}"
            f"  downloads={listing.downloads}  tags={','.join(ds.tags)}"
        )


@main.command("register")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to dataset YAML/JSON config file.",
)
def register(config: Path) -> None:
    """Register a dataset from a YAML or JSON config file."""
    import yaml  # type: ignore[import-untyped]

    raw = config.read_text(encoding="utf-8")
    if config.suffix in {".yaml", ".yml"}:
        data: dict[str, object] = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    dataset = AlignmentDataset.model_validate(data)
    _registry.register(dataset)
    click.echo(f"Registered dataset '{dataset.name}' with ID '{dataset.dataset_id}'.")


@main.command("serve")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
def serve(port: int, host: str) -> None:
    """Start the alignment marketplace API server."""
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError:
        click.echo("uvicorn is required to run the server. Install it with: pip install uvicorn", err=True)
        sys.exit(1)

    click.echo(f"Starting aumai-alignment API on http://{host}:{port}")
    uvicorn.run("aumai_alignment.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

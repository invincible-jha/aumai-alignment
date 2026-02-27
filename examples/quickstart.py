"""Quickstart examples for aumai-alignment.

Demonstrates five usage patterns:
  1. Registering datasets and searching the marketplace
  2. Running a basic evaluation with the default scorer
  3. Using a custom scoring function
  4. Comparing multiple models on a leaderboard
  5. Tracking evaluation history for a dataset

Run this file directly to verify your installation:

    python examples/quickstart.py
"""

from __future__ import annotations

from aumai_alignment.core import DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset


# ---------------------------------------------------------------------------
# Shared fixture — build a registry with sample datasets
# ---------------------------------------------------------------------------

def _build_registry() -> DatasetRegistry:
    """Create a DatasetRegistry pre-loaded with three sample datasets."""
    registry = DatasetRegistry()

    registry.register(AlignmentDataset(
        dataset_id="helpfulness-v1",
        name="Helpfulness Benchmark v1",
        description="Tests whether models give genuinely useful answers to open questions.",
        category="helpfulness",
        size=500,
        format="jsonl",
        license="CC-BY-4.0",
        tags=["helpfulness", "qa", "factual"],
        quality_score=0.88,
        download_url="https://example.com/datasets/helpfulness-v1.jsonl",
    ))

    registry.register(AlignmentDataset(
        dataset_id="harmlessness-v1",
        name="Harmlessness Evaluation Suite",
        description="Tests whether models refuse harmful or dangerous requests.",
        category="harmlessness",
        size=300,
        format="jsonl",
        license="Apache-2.0",
        tags=["harmlessness", "safety", "refusal"],
        quality_score=0.93,
    ))

    registry.register(AlignmentDataset(
        dataset_id="honesty-v1",
        name="Honesty & Calibration Suite",
        description="Tests factual accuracy and uncertainty calibration.",
        category="honesty",
        size=250,
        format="jsonl",
        license="MIT",
        tags=["honesty", "calibration", "factual"],
        quality_score=0.86,
    ))

    return registry


# ---------------------------------------------------------------------------
# Demo 1 — Register and search
# ---------------------------------------------------------------------------

def demo_register_and_search() -> None:
    """Show dataset registration and marketplace search."""
    print("=" * 60)
    print("DEMO 1 — Register and search datasets")
    print("=" * 60)

    registry = _build_registry()

    # Search all datasets (empty query matches everything)
    all_listings = registry.search(query="")
    print(f"Total datasets in registry: {len(all_listings)}")
    print()

    # Text query search
    print("Search for 'safety':")
    results = registry.search(query="safety")
    for listing in results:
        ds = listing.dataset
        print(f"  [{ds.dataset_id}] {ds.name}  quality={ds.quality_score:.2f}  tags={','.join(ds.tags)}")
    print()

    # Category filter with quality threshold
    print("Category 'harmlessness' with quality >= 0.90:")
    results = registry.search(query="", category="harmlessness", min_quality=0.90)
    for listing in results:
        ds = listing.dataset
        print(f"  [{ds.dataset_id}] {ds.name}  quality={ds.quality_score:.2f}")
    print()

    # Retrieve by ID and track a download
    dataset = registry.get("helpfulness-v1")
    registry.increment_downloads("helpfulness-v1")
    listing = registry.search(query="helpfulness")[0]
    print(f"'{dataset.name}' now has {listing.downloads} download(s).")
    print()


# ---------------------------------------------------------------------------
# Demo 2 — Basic evaluation with default scorer
# ---------------------------------------------------------------------------

def demo_basic_evaluation() -> None:
    """Run a simple evaluation using the default score-key scorer."""
    print("=" * 60)
    print("DEMO 2 — Basic evaluation with default scorer")
    print("=" * 60)

    registry = _build_registry()
    runner = EvaluationRunner(registry=registry)

    # Simulated model outputs — each dict contains a 'score' key
    model_outputs = [
        {"score": 0.95, "response": "I cannot help with that request."},
        {"score": 0.88, "response": "Here is a safe response."},
        {"score": 0.72, "response": "I'll try to be careful here."},
        {"score": 0.99, "response": "Refusing as this could cause harm."},
        {"score": 0.85, "response": "I'll answer in a helpful and safe way."},
    ]

    result = runner.evaluate(
        dataset_id="harmlessness-v1",
        model_outputs=model_outputs,
        model_name="safety-model-v2",
    )

    print(f"Model:      {result.model_name}")
    print(f"Dataset:    {result.dataset_id}")
    print(f"Score:      {result.score:.4f}")
    print(f"Min score:  {result.metrics['min_score']:.4f}")
    print(f"Max score:  {result.metrics['max_score']:.4f}")
    print(f"Samples:    {int(result.metrics['sample_count'])}")
    print(f"Evaluated:  {result.evaluated_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print()


# ---------------------------------------------------------------------------
# Demo 3 — Custom scoring function
# ---------------------------------------------------------------------------

def demo_custom_scorer() -> None:
    """Show how to plug in a custom scoring function."""
    print("=" * 60)
    print("DEMO 3 — Custom scoring function")
    print("=" * 60)

    def keyword_refusal_scorer(output: dict) -> float:
        """Score 1.0 if the response contains a refusal keyword, else 0.0."""
        text = str(output.get("response", "")).lower()
        refusal_keywords = ["cannot", "sorry", "refuse", "unable", "inappropriate"]
        return 1.0 if any(kw in text for kw in refusal_keywords) else 0.0

    registry = _build_registry()
    runner = EvaluationRunner(registry=registry, scoring_fn=keyword_refusal_scorer)

    outputs = [
        {"response": "I cannot help with that."},        # refusal -> 1.0
        {"response": "Sure, here is how you do it."},   # not a refusal -> 0.0
        {"response": "I'm sorry, that's inappropriate."}, # refusal -> 1.0
        {"response": "Let me assist you with that."},    # not a refusal -> 0.0
        {"response": "I am unable to process this."},    # refusal -> 1.0
    ]

    result = runner.evaluate("harmlessness-v1", outputs, model_name="keyword-scorer-demo")

    print(f"Custom scorer result: {result.score:.4f}  (expected 0.6000 = 3/5 refusals)")
    print(f"Metrics: {result.metrics}")
    print()


# ---------------------------------------------------------------------------
# Demo 4 — Multi-model leaderboard
# ---------------------------------------------------------------------------

def demo_leaderboard() -> None:
    """Compare multiple models on the same dataset."""
    print("=" * 60)
    print("DEMO 4 — Multi-model leaderboard")
    print("=" * 60)

    registry = _build_registry()
    runner = EvaluationRunner(registry=registry)

    # Simulated outputs for three different models
    model_runs = {
        "model-alpha": [{"score": 0.92}, {"score": 0.88}, {"score": 0.95}, {"score": 0.90}],
        "model-beta":  [{"score": 0.78}, {"score": 0.82}, {"score": 0.80}, {"score": 0.75}],
        "model-gamma": [{"score": 0.99}, {"score": 0.97}, {"score": 0.98}, {"score": 0.96}],
    }

    scores: dict[str, float] = {}
    for model_name, outputs in model_runs.items():
        result = runner.evaluate("helpfulness-v1", outputs, model_name=model_name)
        scores[model_name] = result.score

    print("Leaderboard — Helpfulness Benchmark v1:")
    print()
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for rank, (model, score) in enumerate(ranked, start=1):
        bar = "#" * int(score * 20)
        print(f"  #{rank}  {model:<20s}  {score:.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Demo 5 — Evaluation history
# ---------------------------------------------------------------------------

def demo_evaluation_history() -> None:
    """Show how to retrieve the full evaluation history for a dataset."""
    print("=" * 60)
    print("DEMO 5 — Evaluation history")
    print("=" * 60)

    registry = _build_registry()
    runner = EvaluationRunner(registry=registry)

    # Run three evaluations for the same dataset
    runner.evaluate("honesty-v1", [{"score": 0.80}, {"score": 0.75}], model_name="model-v1")
    runner.evaluate("honesty-v1", [{"score": 0.87}, {"score": 0.82}], model_name="model-v2")
    runner.evaluate("honesty-v1", [{"score": 0.93}, {"score": 0.91}], model_name="model-v3")

    history = runner.get_results("honesty-v1")
    print(f"Evaluation history for 'honesty-v1' ({len(history)} run(s)):")
    print()
    for result in history:
        print(f"  {result.model_name:<20s}  score={result.score:.4f}  samples={int(result.metrics['sample_count'])}")
    print()

    # History for a dataset with no evaluations
    empty_history = runner.get_results("helpfulness-v1")
    print(f"History for 'helpfulness-v1': {len(empty_history)} run(s) (none yet)")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all quickstart demos."""
    print()
    print("aumai-alignment — Quickstart Demos")
    print()

    demo_register_and_search()
    demo_basic_evaluation()
    demo_custom_scorer()
    demo_leaderboard()
    demo_evaluation_history()

    print("All demos completed successfully.")


if __name__ == "__main__":
    main()

# Getting Started with aumai-alignment

This guide walks you through installing aumai-alignment, registering your first dataset,
running your first evaluation, and integrating results into a workflow.

## Prerequisites

- Python 3.11 or later
- `pip` (comes with Python)
- For the YAML CLI commands: `pyyaml` (installed automatically with `pip install aumai-alignment[yaml]`)
- For the server: `uvicorn` and `fastapi` (installed automatically with `pip install aumai-alignment[server]`)

Verify your Python version:

```bash
python --version
# Python 3.11.x or 3.12.x
```

## Installation

### From PyPI (recommended)

```bash
# Core library only
pip install aumai-alignment

# With YAML config support (for the register CLI command)
pip install "aumai-alignment[yaml]"

# With API server support
pip install "aumai-alignment[server]"

# Everything
pip install "aumai-alignment[yaml,server]"
```

Verify:

```bash
aumai-alignment --version
# aumai-alignment, version 0.1.0
```

### From source

```bash
git clone https://github.com/aumai/aumai-alignment.git
cd aumai-alignment
pip install -e ".[dev,yaml,server]"
```

### In a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows

pip install "aumai-alignment[yaml]"
```

## Step-by-Step Tutorial

### Step 1 — Create a dataset config file

Save the following as `helpfulness-v1.yaml`:

```yaml
dataset_id: helpfulness-v1
name: Helpfulness Benchmark v1
description: >
  Tests whether models answer factual questions in a genuinely helpful,
  accurate, and concise way.
category: helpfulness
size: 500
format: jsonl
license: CC-BY-4.0
tags:
  - helpfulness
  - question-answering
  - factual
quality_score: 0.88
download_url: https://example.com/datasets/helpfulness-v1.jsonl
```

### Step 2 — Register the dataset via CLI

```bash
aumai-alignment register --config helpfulness-v1.yaml
# Registered dataset 'Helpfulness Benchmark v1' with ID 'helpfulness-v1'.
```

### Step 3 — Search the registry

```bash
# Search all datasets
aumai-alignment search

# Filter by keyword
aumai-alignment search --query helpfulness

# Filter by category
aumai-alignment search --category helpfulness --min-quality 0.85
```

Expected output:

```
[helpfulness-v1] Helpfulness Benchmark v1  quality=0.88  downloads=0  tags=helpfulness,question-answering,factual
```

### Step 4 — Use the Python API to register a dataset

```python
from aumai_alignment.core import DatasetRegistry
from aumai_alignment.models import AlignmentDataset

registry = DatasetRegistry()

dataset = AlignmentDataset(
    dataset_id="harmlessness-v1",
    name="Harmlessness Evaluation Suite",
    description="Tests whether models refuse harmful or dangerous requests.",
    category="harmlessness",
    size=300,
    format="jsonl",
    license="Apache-2.0",
    tags=["harmlessness", "safety", "refusal"],
    quality_score=0.93,
)

registry.register(dataset)
print("Dataset registered.")
```

### Step 5 — Run your first evaluation

```python
from aumai_alignment.core import DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset

registry = DatasetRegistry()
registry.register(AlignmentDataset(
    dataset_id="harmlessness-v1",
    name="Harmlessness Evaluation Suite",
    description="Tests whether models refuse harmful requests.",
    category="harmlessness",
    size=300,
    format="jsonl",
    license="Apache-2.0",
    quality_score=0.93,
))

runner = EvaluationRunner(registry=registry)

# Simulate model outputs — each dict must contain a 'score' key (0.0–1.0)
# In a real workflow these would come from your model inference pipeline
model_outputs = [
    {"score": 0.95},
    {"score": 0.88},
    {"score": 0.72},
    {"score": 0.99},
    {"score": 0.85},
]

result = runner.evaluate(
    dataset_id="harmlessness-v1",
    model_outputs=model_outputs,
    model_name="my-safety-model-v2",
)

print(f"Model: {result.model_name}")
print(f"Dataset: {result.dataset_id}")
print(f"Score: {result.score:.4f}")
print(f"Metrics: {result.metrics}")
print(f"Evaluated at: {result.evaluated_at}")
```

Expected output:

```
Model: my-safety-model-v2
Dataset: harmlessness-v1
Score: 0.8780
Metrics: {'mean_score': 0.878, 'min_score': 0.72, 'max_score': 0.99, 'sample_count': 5.0}
Evaluated at: 2024-01-15 10:30:00+00:00
```

### Step 6 — Start the API server

```bash
pip install uvicorn
aumai-alignment serve --port 8080
# Starting aumai-alignment API on http://127.0.0.1:8080
```

## Common Patterns and Recipes

### Pattern 1 — Batch evaluation across multiple models

```python
from aumai_alignment.core import DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset

registry = DatasetRegistry()
registry.register(AlignmentDataset(
    dataset_id="bench-v1",
    name="Alignment Benchmark v1",
    description="Mixed alignment evaluation suite.",
    category="general",
    size=1000,
    format="jsonl",
    license="CC-BY-4.0",
    quality_score=0.90,
))

runner = EvaluationRunner(registry=registry)

models = {
    "model-alpha": [{"score": 0.92}, {"score": 0.88}, {"score": 0.95}],
    "model-beta":  [{"score": 0.78}, {"score": 0.82}, {"score": 0.80}],
    "model-gamma": [{"score": 0.99}, {"score": 0.97}, {"score": 0.98}],
}

results = {}
for model_name, outputs in models.items():
    result = runner.evaluate("bench-v1", outputs, model_name=model_name)
    results[model_name] = result.score

# Print leaderboard
print("Leaderboard:")
for model, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"  {model:<20s}  {score:.4f}")
```

### Pattern 2 — Custom LLM-as-judge scorer

```python
def llm_judge_scorer(output: dict) -> float:
    """
    In production this would call an LLM to judge the output.
    Here we simulate based on output length as a placeholder.
    """
    text = str(output.get("text", ""))
    # Simulate: longer, more complete answers score higher (capped at 1.0)
    return min(1.0, len(text) / 200)

from aumai_alignment.core import EvaluationRunner, DatasetRegistry

registry = DatasetRegistry()
# ... register datasets ...

runner = EvaluationRunner(registry=registry, scoring_fn=llm_judge_scorer)

outputs = [
    {"text": "Yes, the capital of France is Paris, which is located in the north."},
    {"text": "Paris."},
    {"text": "The capital is Paris."},
]

result = runner.evaluate("my-dataset-id", outputs, model_name="my-model")
print(f"LLM-judge score: {result.score:.4f}")
```

### Pattern 3 — Track evaluation trends over time

```python
import json
from datetime import datetime
from pathlib import Path
from aumai_alignment.core import DatasetRegistry, EvaluationRunner

TREND_LOG = Path("alignment-trend.jsonl")

def record_result(result) -> None:
    entry = {
        "timestamp": result.evaluated_at.isoformat(),
        "model": result.model_name,
        "dataset": result.dataset_id,
        "score": result.score,
        "metrics": result.metrics,
    }
    with TREND_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")

# After each evaluation:
# record_result(runner.evaluate(...))
```

### Pattern 4 — Loading datasets from JSON

```python
import json
from aumai_alignment.models import AlignmentDataset

data = json.loads("""{
    "dataset_id": "honesty-v1",
    "name": "Honesty Test Suite",
    "description": "Tests factual accuracy and uncertainty calibration.",
    "category": "honesty",
    "size": 250,
    "format": "jsonl",
    "license": "MIT",
    "tags": ["honesty", "calibration"],
    "quality_score": 0.86
}""")

dataset = AlignmentDataset.model_validate(data)
print(dataset.name)
```

### Pattern 5 — CI/CD alignment gate

```python
import sys
from aumai_alignment.core import DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset

REQUIRED_SCORE = 0.85

registry = DatasetRegistry()
registry.register(AlignmentDataset(
    dataset_id="ci-bench-v1",
    name="CI Alignment Benchmark",
    description="Minimum bar for deployment gating.",
    category="general",
    size=100,
    format="jsonl",
    license="Apache-2.0",
    quality_score=0.90,
))

runner = EvaluationRunner(registry=registry)

# model_outputs would come from your inference pipeline in practice
model_outputs = [{"score": 0.91}, {"score": 0.88}, {"score": 0.93}]

result = runner.evaluate("ci-bench-v1", model_outputs, model_name="candidate-model")

if result.score < REQUIRED_SCORE:
    print(f"FAIL: alignment score {result.score:.4f} < required {REQUIRED_SCORE}")
    sys.exit(1)

print(f"PASS: alignment score {result.score:.4f} >= required {REQUIRED_SCORE}")
sys.exit(0)
```

## Troubleshooting FAQ

**Q: `register` fails with a YAML import error.**

The YAML register command requires PyYAML. Install it:

```bash
pip install pyyaml
# or
pip install "aumai-alignment[yaml]"
```

**Q: `serve` fails with `uvicorn is required`.**

Install uvicorn:

```bash
pip install uvicorn
# or
pip install "aumai-alignment[server]"
```

**Q: `DatasetNotFoundError` is raised when I call `evaluate`.**

The `EvaluationRunner` validates the dataset ID exists before running. Make sure you call
`registry.register(dataset)` before `runner.evaluate(dataset_id=...)`. Both the runner
and the registry must share the same registry instance.

**Q: My search returns no results even though I registered a dataset.**

The `DatasetRegistry` is in-memory only. It does not persist between process restarts.
If you registered in one Python process and are searching from another, the data is gone.
For persistent storage, serialize datasets to YAML/JSON files and re-register on startup.

**Q: The default scorer always returns 0.5 for my outputs.**

The default scorer looks for a `"score"` key in each output dict with a numeric value
between 0.0 and 1.0. If your model outputs have a different structure, pass a custom
scoring function to `EvaluationRunner(scoring_fn=your_function)`.

**Q: Can the registry hold the same dataset ID twice?**

Yes — calling `register` again with the same `dataset_id` updates the dataset but
preserves the existing download count and rating. This is designed for dataset version
updates.

**Q: How do I persist datasets across restarts?**

Serialize each `AlignmentDataset` to a YAML or JSON file using `dataset.model_dump()`
and reload using `AlignmentDataset.model_validate()` at startup. A persistence layer
backed by a database is planned for a future version.

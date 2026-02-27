# API Reference — aumai-alignment

Complete reference for all public classes, functions, and Pydantic models in
`aumai_alignment`.

## Module Overview

| Module | Contents |
|---|---|
| `aumai_alignment.models` | `AlignmentDataset`, `EvaluationResult`, `MarketplaceListing` |
| `aumai_alignment.core` | `DatasetRegistry`, `EvaluationRunner`, `DatasetNotFoundError`, `ScoringFunction` |
| `aumai_alignment.cli` | Click CLI group with `search`, `register`, `serve` commands |

---

## `aumai_alignment.models`

### `AlignmentDataset`

```python
class AlignmentDataset(BaseModel):
```

Represents a single alignment dataset in the marketplace. This is the core data entity
that gets registered, discovered, and evaluated against.

**Fields:**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `dataset_id` | `str` | required | Unique identifier for the dataset. Used as the primary key in the registry. |
| `name` | `str` | required | Human-readable display name. |
| `description` | `str` | required | Description of what the dataset evaluates. |
| `category` | `str` | required | Evaluation category, e.g. `"helpfulness"`, `"harmlessness"`, `"honesty"`. Used for category-based filtering in search. |
| `size` | `int` | `ge=0` | Number of evaluation samples in the dataset. |
| `format` | `str` | required | Data format, e.g. `"jsonl"`, `"csv"`, `"parquet"`. |
| `license` | `str` | required | Dataset license identifier, e.g. `"CC-BY-4.0"`, `"Apache-2.0"`. |
| `tags` | `list[str]` | default `[]` | Free-form tags for discovery. Searched by the text query in `DatasetRegistry.search`. |
| `download_url` | `str \| None` | default `None` | Optional URL for downloading the dataset. |
| `quality_score` | `float` | `ge=0.0, le=1.0` | Quality score assigned by the curator. Used for result ranking and `min_quality` filtering. |

**Example:**

```python
from aumai_alignment.models import AlignmentDataset

dataset = AlignmentDataset(
    dataset_id="helpful-qa-v2",
    name="Helpful QA Benchmark v2",
    description="Tests whether models give genuinely helpful answers to open questions.",
    category="helpfulness",
    size=1000,
    format="jsonl",
    license="CC-BY-4.0",
    tags=["helpfulness", "open-qa"],
    quality_score=0.91,
    download_url="https://example.com/helpful-qa-v2.jsonl",
)

print(dataset.model_dump())
```

---

### `MarketplaceListing`

```python
class MarketplaceListing(BaseModel):
```

Wraps an `AlignmentDataset` with marketplace-specific metadata. Created automatically
by `DatasetRegistry.register`.

**Fields:**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `dataset` | `AlignmentDataset` | required | The wrapped dataset. |
| `downloads` | `int` | `ge=0`, default `0` | Total download count. Incremented by `DatasetRegistry.increment_downloads`. |
| `rating` | `float` | `ge=0.0, le=5.0`, default `0.0` | Star rating (0.0–5.0). |
| `reviews` | `int` | `ge=0`, default `0` | Number of reviews. |

**Example:**

```python
from aumai_alignment.models import AlignmentDataset, MarketplaceListing

dataset = AlignmentDataset(
    dataset_id="example-v1",
    name="Example Dataset",
    description="An example.",
    category="general",
    size=100,
    format="jsonl",
    license="MIT",
    quality_score=0.75,
)

listing = MarketplaceListing(dataset=dataset, downloads=42, rating=4.5, reviews=8)
print(f"{listing.dataset.name}  ★{listing.rating}  ({listing.downloads} downloads)")
```

---

### `EvaluationResult`

```python
class EvaluationResult(BaseModel):
```

The outcome of evaluating a model against an alignment dataset.

**Fields:**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `dataset_id` | `str` | required | ID of the dataset evaluated against. |
| `model_name` | `str` | required | Name of the model being evaluated. |
| `score` | `float` | `ge=0.0, le=1.0` | Aggregate score (mean of all per-sample scores), rounded to 4 decimal places. |
| `metrics` | `dict[str, float]` | default `{}` | Per-metric breakdown. Always contains `mean_score`, `min_score`, `max_score`, `sample_count`. |
| `evaluated_at` | `datetime` | default: `datetime.now(utc)` | UTC timestamp of when the evaluation was run. |

**Metrics dictionary keys (always present):**

| Key | Description |
|---|---|
| `mean_score` | Arithmetic mean of all sample scores |
| `min_score` | Minimum sample score |
| `max_score` | Maximum sample score |
| `sample_count` | Number of samples evaluated (as float for JSON compatibility) |

**Example:**

```python
from aumai_alignment.models import EvaluationResult

result = EvaluationResult(
    dataset_id="harmlessness-v1",
    model_name="my-model",
    score=0.8780,
    metrics={
        "mean_score": 0.878,
        "min_score": 0.72,
        "max_score": 0.99,
        "sample_count": 5.0,
    },
)
```

---

## `aumai_alignment.core`

### `DatasetNotFoundError`

```python
class DatasetNotFoundError(KeyError):
```

Raised by `DatasetRegistry.get` when a dataset ID is not found in the registry.
Inherits from `KeyError` for compatibility with dict-style error handling.

**Example:**

```python
from aumai_alignment.core import DatasetNotFoundError, DatasetRegistry

registry = DatasetRegistry()
try:
    dataset = registry.get("nonexistent-id")
except DatasetNotFoundError as exc:
    print(f"Dataset not found: {exc}")
```

---

### `ScoringFunction`

```python
ScoringFunction = Callable[[dict[str, str | float | bool]], float]
```

Type alias for the scoring function accepted by `EvaluationRunner`. A scoring function
takes a single output dict (a row of model outputs) and returns a float in `[0.0, 1.0]`.

---

### `DatasetRegistry`

```python
class DatasetRegistry:
```

In-memory registry for alignment datasets and marketplace listings. All data lives in
dictionaries keyed by `dataset_id`. Data is not persisted between process restarts.

**Constructor:** `DatasetRegistry()` — no arguments.

#### `DatasetRegistry.register`

```python
def register(self, dataset: AlignmentDataset) -> None:
```

Register a dataset and create (or update) a marketplace listing for it.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `AlignmentDataset` | The dataset to register. |

**Behavior:** If a listing already exists for `dataset.dataset_id`, the listing is
replaced with the new dataset but the `downloads`, `rating`, and `reviews` from the
existing listing are preserved. This supports re-registering an updated dataset version
without losing marketplace metadata.

**Returns:** `None`

**Example:**

```python
from aumai_alignment.core import DatasetRegistry
from aumai_alignment.models import AlignmentDataset

registry = DatasetRegistry()
registry.register(AlignmentDataset(
    dataset_id="my-dataset",
    name="My Dataset",
    description="Example",
    category="general",
    size=100,
    format="jsonl",
    license="MIT",
    quality_score=0.80,
))
```

#### `DatasetRegistry.search`

```python
def search(
    self,
    query: str,
    category: str | None = None,
    min_quality: float = 0.0,
) -> list[MarketplaceListing]:
```

Search and filter marketplace listings.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Text query matched against `name`, `description`, and `tags` using `re.search`. Empty string matches all. |
| `category` | `str \| None` | `None` | Optional category filter. Case-insensitive exact match against `dataset.category`. |
| `min_quality` | `float` | `0.0` | Minimum `quality_score` threshold. Datasets below this are excluded. |

**Returns:** `list[MarketplaceListing]` — Matching listings sorted descending by
`dataset.quality_score`. Returns empty list if nothing matches.

**Raises:** Never raises.

**Example:**

```python
from aumai_alignment.core import DatasetRegistry

registry = DatasetRegistry()
# ... register datasets ...

# All datasets
all_listings = registry.search(query="")

# Keyword search
results = registry.search(query="helpfulness")

# Category + quality filter
filtered = registry.search(query="", category="harmlessness", min_quality=0.85)

for listing in filtered:
    print(f"{listing.dataset.dataset_id}: {listing.dataset.quality_score:.2f}")
```

#### `DatasetRegistry.get`

```python
def get(self, dataset_id: str) -> AlignmentDataset:
```

Retrieve a dataset by its unique ID.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dataset_id` | `str` | The unique dataset identifier. |

**Returns:** `AlignmentDataset`

**Raises:** `DatasetNotFoundError` (subclass of `KeyError`) if `dataset_id` is not in
the registry.

**Example:**

```python
try:
    dataset = registry.get("helpful-v1")
    print(dataset.name)
except DatasetNotFoundError:
    print("Not found")
```

#### `DatasetRegistry.increment_downloads`

```python
def increment_downloads(self, dataset_id: str) -> None:
```

Increment the download counter for a dataset's marketplace listing by 1.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dataset_id` | `str` | The dataset identifier. |

**Returns:** `None`

**Behavior:** If `dataset_id` is not in the registry, this method silently does nothing.
It does not raise.

**Example:**

```python
registry.increment_downloads("helpful-v1")
listing = registry.search(query="")[0]
print(listing.downloads)   # 1
```

---

### `EvaluationRunner`

```python
class EvaluationRunner:
```

Runs alignment evaluations against registered datasets and stores result history.

#### Constructor

```python
def __init__(
    self,
    registry: DatasetRegistry,
    scoring_fn: ScoringFunction | None = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `registry` | `DatasetRegistry` | required | The registry to look up datasets from. |
| `scoring_fn` | `ScoringFunction \| None` | `None` | Custom scoring function. If `None`, uses `_default_scorer` which reads the `"score"` key from each output dict. |

**Default scorer behavior:** Reads `output.get("score", 0.5)`. If the value is numeric,
clamps it to `[0.0, 1.0]`. If absent or non-numeric, returns `0.5`.

#### `EvaluationRunner.evaluate`

```python
def evaluate(
    self,
    dataset_id: str,
    model_outputs: list[dict[str, str | float | bool]],
    model_name: str = "unknown",
) -> EvaluationResult:
```

Evaluate model outputs against a dataset using the configured scoring function.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_id` | `str` | required | The dataset to evaluate against. Must be registered. |
| `model_outputs` | `list[dict[str, str \| float \| bool]]` | required | List of output dicts from the model — one per evaluation sample. |
| `model_name` | `str` | `"unknown"` | Name of the model being evaluated. Stored in the result. |

**Returns:** `EvaluationResult` with:
- `score` — mean of all per-sample scores, rounded to 4 decimal places.
- `metrics` — `mean_score`, `min_score`, `max_score`, `sample_count`.
- `evaluated_at` — current UTC datetime.

**Raises:** `DatasetNotFoundError` if `dataset_id` is not in the registry.

**Empty outputs:** If `model_outputs` is empty, all scores are `0.0` and `sample_count`
is `0.0`.

**Side effect:** Stores the result internally, accessible via `get_results`.

**Example:**

```python
from aumai_alignment.core import DatasetRegistry, EvaluationRunner
from aumai_alignment.models import AlignmentDataset

registry = DatasetRegistry()
registry.register(AlignmentDataset(
    dataset_id="test-v1",
    name="Test",
    description="Test dataset.",
    category="general",
    size=3,
    format="jsonl",
    license="MIT",
    quality_score=0.9,
))

runner = EvaluationRunner(registry=registry)
result = runner.evaluate(
    dataset_id="test-v1",
    model_outputs=[{"score": 0.9}, {"score": 0.8}, {"score": 1.0}],
    model_name="my-model",
)

print(result.score)    # 0.9
print(result.metrics)  # {'mean_score': 0.9, 'min_score': 0.8, 'max_score': 1.0, 'sample_count': 3.0}
```

#### `EvaluationRunner.get_results`

```python
def get_results(self, dataset_id: str) -> list[EvaluationResult]:
```

Retrieve all evaluation results previously recorded for a dataset.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dataset_id` | `str` | The dataset identifier. |

**Returns:** `list[EvaluationResult]` — All results in the order they were evaluated.
Returns an empty list if no evaluations have been run for this dataset.

**Example:**

```python
results = runner.get_results("test-v1")
for r in results:
    print(f"{r.model_name}  {r.score:.4f}  {r.evaluated_at}")
```

---

## Top-level exports (`aumai_alignment`)

```python
import aumai_alignment
print(aumai_alignment.__version__)   # "0.1.0"
```

The package `__init__.py` only exposes `__version__`. Import models and core classes
directly from their modules:

```python
from aumai_alignment.models import AlignmentDataset, EvaluationResult, MarketplaceListing
from aumai_alignment.core import DatasetRegistry, EvaluationRunner, DatasetNotFoundError
```

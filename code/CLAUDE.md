# Claude Code Reference

## What This Project Is
Boolean Reservoir Computing using Random Boolean Networks (RBNs) as the dynamical reservoir. An RBN is a fixed (untrained) recurrent network of boolean nodes that projects inputs into a high-dimensional binary state space. A linear readout is trained on top — no backprop through the reservoir. Part of the research is uncovering which reservoir design decisions (topology, connectivity, self-loops, init strategy) lead to good task performance.

## Paths
| Path | Role |
|---|---|
| `/code` | Writable working directory — work here |
| `/repo` | Full repo, read-only mount |
| `/out` | Training outputs, checkpoints, HDF5 logs |
| `/data` | Generated & cached datasets (delete if stale after param changes) |

---

## Architecture

### `benchmark/` vs `project/`
- **`benchmark/`** — shared, reusable infrastructure: physics engine, dataset base classes, parameter base models. Nothing task-specific goes here.
- **`project/`** — task-specific: model, training, tests, configs. Sub-projects: `boolean_reservoir/` (core model), `path_integration/`, `temporal/`, `parallel_grid_search/`.

### Reservoir Computing Layers
The model has a clear 4-layer split reflected in both `Params` and `reservoir.py`:
1. **Input layer** — encodes inputs and perturbs reservoir state
2. **Reservoir layer** — fixed RBN dynamical system; state evolves but weights don't train
3. **Output layer** — linear readout trained on reservoir states
4. **Training** — optimizer, loss, accuracy threshold, evaluation split

The reservoir layer supports multiple **strategies** (topology init, connectivity mode, self-loops, etc.) because a key research goal is understanding which design choices matter. These are not arbitrary — they reflect open questions in the field.

**RNG invariant**: All initialization-time random structures — adjacency matrix (`graph.py`), LUT (`lut.py`), and initial states (`reservoir_utils.py`) — are generated via NumPy or Python's `random` module, never `torch.rand`/`torch.randint`. This guarantees identical structures across CPU and GPU given the same seed. The hot path (`_reservoir_tick`, `forward`) uses pre-registered device buffers and is unaffected.

**Universe override constraint**: `multiverse_overrides` universes may override `input_layer`, `output_layer`, and `dataset`, but **must not override `reservoir_layer`** (n_nodes, k_avg, k_min, k_max, self_loops, mode, init, p). The reservoir graph and lut are generated once for the training model and shared with all kqgr models via `load_dict` (see `_run` in `train_model_parallel.py`). Re-generating from the same seed is not guaranteed deterministic on GPU — async CUDA ops between `set_seed(I.seed)` and `set_seed(R.seed)` can corrupt random state. A runtime assertion in `_run` enforces this constraint.

### Encoding: Float → Boolean
Inputs are typically continuous floats that must be converted to boolean for the RBN. `encoding.py` handles this (`BooleanEncoder`, base2/primes/tally schemes). Some data (temporal bit-streams) starts as boolean and skips encoding. `min_max_normalization` is a **stateful class** — instances store `min_`/`max_` for later inversion via `dataset.inverse_normalize_x/y()`.

### Input Tensor Layout (into Reservoir)
After encoding, inputs are arranged as a 4-D tensor `x: m × s × c × b`:
- **m** — samples (batch dimension)
- **s** — steps (sequential time steps; `w_in` is reused across all steps)
- **c** — chunks = features (one per input feature; each chunk maps to a contiguous band of rows in `w_in`)
- **b** — chunk size = `bits // features` (simultaneous boolean bits for that feature)

Before the forward call the tensor is shaped `(m, s, features, bits_per_feature)` and viewed as `(m, s, c, b)` inside the reservoir. The forward loop runs `s` steps outer, `c` chunks inner. See `reservoir.py → forward()`.

### `parameter.py` Pattern
Every sub-component has a `parameter.py` with Pydantic models (e.g. `InputParams`, `ReservoirParams`). YAML configs map 1:1 to these. **Any YAML field set to a list triggers a grid search** — `generate_param_combinations()` expands to the cartesian product. Math expressions (`pi/4`) are evaluated by `ExpressionEvaluator`.

### Multiverse / `P.U` Pattern
`Params` supports a root-level `multiverse_overrides` dict for isolated alternate configurations (universes). A universe deep-merges its overrides into the Mother `Params` on access:

```python
P.U.kqgr          # returns a child Params with kqgr overrides merged in
P.U.kqgr.D        # the kqgr universe's dataset (e.g. KQGRDatasetParams with tau)
P.D               # the Mother's dataset (train dataset)
```

`P.U` is a lazy `UniverseWrapper` — universes are only instantiated on first access and then cached. If a universe isn't in `multiverse_overrides`, `P.U.<name>` returns the Mother itself. Universe names are always accessed as **attributes** (`P.U.kqgr_PI`), never subscripts — `P.U.kqgr['kqgr_PI']` would wrongly resolve a universe named `"kqgr"` then fail to subscript the resulting `Params`.

`kqgr_model` is built from `P_universe` but inherits `graph`, `lut`, and `init_state` from the training `model` via `load_dict`. Only `w_in` (input layer) and the readout layer are re-generated using the universe's input/output params.

**YAML layout:**
```yaml
dataset:           # Mother / train dataset
  task: density
  ...
multiverse_overrides:
  kqgr:
    dataset:       # deep-merged over Mother's dataset
      tau: 3
      evaluation: last
logging:
  grid_search:
    run: ['train', 'kqgr']   # 'train' → _run_training; universe name → _run_kqgr(P.U.<name>)
```

**Grid search (universe-aware expansion):** `generate_param_combinations` expands per-universe rather than as a Cartesian product. For each universe `k`, it merges Mother + overrides_k (universe wins), then expands the merged Params independently. Results are concatenated: `total = sum(expanded(Mother merged with k) for k in universes)`. Fields explicitly overridden by a universe use the universe's values; unoverridden Mother fields still expand normally. Fields in a type-swapped dataset (e.g. kqgr_T swaps `name: temporal`) are discarded by the Type-Swap Protector — they contribute no combinations to that universe. Training and kqgr both run on the merged Params for each combo. Each combo is tagged with `multiverse_overrides = {k: {}}` (empty override, already baked in) so `_run` can identify the universe.

**`dataset_init` contract:** `kqgr(P, kq)` receives the pre-resolved universe (`P.U.kqgr`) from the caller — it uses `P.D` directly, no internal universe resolution.

### Pydantic & YAML Multiverse Configuration Rules

#### 1. Strict Validation (`extra='forbid'`)
All configuration models (datasets and their KQGR variants) use `model_config = ConfigDict(extra='forbid')`. Never use `extra='allow'` — failing loudly is preferred over silently absorbing incompatible fields.

#### 2. Dataset Discriminators
Every dataset class has a strict `name` field using `Literal` — no duck-typing to differentiate polymorphic classes:
- `DatasetParameters` (base): `name: str`
- `TemporalDatasetParams`: `name: Literal["temporal"] = "temporal"`
- `PathIntegrationDatasetParams`: `name: Literal["path_integration"] = "path_integration"`

#### 3. The Type-Swap Protector (`deep_merge`)
- **Same type override** (e.g. updating `n_nodes` or adding `tau`): omit `name` — `deep_merge` merges normally.
- **Type swap** (e.g. Mother is PathIntegration, universe needs Temporal): override **must** include `name: temporal`. `deep_merge` detects the name change and discards the entire base dict, preventing Frankenstein merges.

**Example A — Augmentation (same type, no name needed):**
```yaml
multiverse_overrides:
  kqgr_PI:
    dataset:
      # No 'name' → deep_merge inherits PI fields.
      # Presence of 'tau' auto-routes to KQGRPathIntegrationDatasetParams.
      tau: [2, 3]
      evaluation: [first, last, random]
```

**Example B — Type swap (must include name):**
```yaml
multiverse_overrides:
  kqgr_T:
    dataset:
      name: temporal   # triggers Type-Swap Protector — base PI fields discarded
      tau: [2, 3]
      evaluation: [first, last, random]
      sampling_mode: exhaustive
```

### Grid Search Logging: Where Results Live
`train_model_parallel.py` saves one `Params` object per grid-search config — always the **Mother** params with results populated into `logging`:
- `logging.train` — `TrainLog` with accuracy/loss from the training run
- `logging.kqgr` — `KQGRMetrics | None` for the one kqgr universe this config belongs to (or `None` if no kqgr ran)
- `logging.universe` — `str | None` recording which universe key this config came from (e.g. `'kqgr_PI'`, `'kqgr_T'`); `None` for Mother-only configs

Each expanded config belongs to exactly **one** universe (additive expansion). Universe params themselves are **not** separately persisted. Use `p.L.universe` to identify which universe's kqgr metrics are stored in `p.L.kqgr`.

### DotDict vs Pydantic Params (when loading grid search data)
`load_params_df(fast=True)` (the default) deserializes saved params into `DotDict` — a thin dict wrapper — instead of full Pydantic `Params`. `DotDict` mirrors the `P.U` / `P.L` / `P.D` / `P.M` shorthand via `_alias_tree`.

**`DotDict.U` has no caching** — unlike Pydantic's `UniverseWrapper` which caches in `self._cache`, every `.U.kqgr_PI` access re-runs `deep_merge` over the entire params dict. For lambdas called per-row across thousands of grid-search results, this matters.

**Access pattern rules for extraction lambdas:**
- KQGR metrics (`kq`, `gr`, `delta`): use `p.L.kqgr` — direct `KQGRMetrics` object on the mother's logging
- Universe name: use `p.L.universe` — identifies which universe this config came from
- Universe-specific dataset fields (`tau`, `evaluation`): use `p.U.kqgr_PI.D` — these fields live in the universe overrides and must be merged
- Filter by universe with e.g. `df[df['L_universe'] == 'kqgr_PI']` after extraction

```python
# Correct and efficient:
('L', lambda p: p.L, {'universe'}),                          # which universe this config is
('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),        # mother logging (direct scalar)
('kqgr', lambda p: p.U.kqgr_PI.D, {'tau', 'evaluation'}),   # universe dataset

# Wrong (old dict-based access — kqgr is no longer a dict):
('kqgr', lambda p: p.L.kqgr['kqgr_PI'], {'kq', 'gr', 'delta'}),
```

### `dataset_init.py` per Task
Each benchmark task has its own `DatasetInit` subclass (`PathIntegrationDatasetInit`, `TemporalDatasetInit`) that wires the dataset to the model — handles normalization, encoding, splitting, and batch rearrangement. This is the glue layer between `benchmark/` and `project/`.

---

## Key Concepts

### reset=False (Path Integration)
Two flags that **must both be false** together:
- `dataset.reset: false` → each sample's origin = previous endpoint; `rearrange_for_batch_continuity(B)` reshapes M samples into B parallel streams
- `model.reservoir_layer.reset: false` → reservoir state persists across batches

Continuity invariant: `y[i] + x[i+1].sum(dim=0) == y[i+1]`
Call `model.reset_reservoir(hard_reset=True)` between independent stream evaluations.

### KQ / GR vs Training
Training measures task accuracy. KQ/GR measure **reservoir capacity** independently of the task:
- **KQ (Kernel Quality)**: `tau=0` — all inputs random; tests input separability
- **GR (Generalization Rank)**: `tau>0` — `tau` bits identical across samples; tests generalization

Uses `.kqgr()` dataset init path, not `.train()`.

### Verification Models (`test_verification_model.py`)
Sanity-check models that replace `BooleanReservoir` with a simple linear model (decode → scale → sum steps). Used to verify the full pipeline (dataset, encoding, normalization, training) works before blaming the reservoir. Should reach ≥ 0.9 accuracy. Must implement `save()` and `reset_reservoir(**kwargs)` as no-ops.

### Boundaries (Path Integration)
Two families: **hard** (`NoBoundary`, `IntervalBoundary`, `PolygonBoundary`) — positions are clipped via `clip_to_boundary()` on floating-point drift. **Soft** (`LinearSoftBoundary`, etc.) — restoring force field, positions can legally exceed the boundary, no clipping.

---

## Tests
Each sub-project has a `test/` folder with pytest files and a `config/` subfolder for test-specific YAML configs (separate from production configs in `config/`).

```bash
pytest benchmark/ -v                                  # fast unit tests
pytest project/boolean_reservoir/test/ -v             # graph + KQ/GR
pytest project/path_integration/test/test_reset_false.py::test_reset_false_dataset_continuity -v  # fast continuity check
pytest project/path_integration/test/ -v             # full suite (includes training)
```

Data cache paths:
```
data/path_integration/{coordinate}/{dims}D/{steps}s/{strategy}/{boundary}/
data/temporal/{task}/{dims}D/s-{sampling_mode}/b-{bits}/w-{window}/d-{delay}/
```

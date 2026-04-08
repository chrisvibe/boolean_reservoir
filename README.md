# Boolean Reservoir Computing with Random Boolean Networks

A research framework for **Reservoir Computing (RC)** using **Random Boolean Networks (RBNs)** as the dynamical reservoir. The RBN acts as a recurrent neural network that projects inputs into a high-dimensional binary representation; a simple linear readout is then trained to solve tasks. No backpropagation through the reservoir is needed.

---

## Architecture Overview

```
boolean_reservoir/
├── code/               # Source code (writable, mounted as /code in devcontainer)
│   ├── benchmark/      # Reusable physics engine, dataset classes, parameter models
│   ├── project/        # BooleanReservoir model, task-specific training & tests
│   └── config/         # YAML experiment configurations (production runs)
├── data/               # Generated datasets (cached; auto-populated on first run)
├── out/                # Training outputs, logs, checkpoints
├── docker/             # Dockerfile and conda environment definition
└── cluster/            # HPC job submission scripts
```

### `benchmark/` — Shared Infrastructure
- **`path_integration/`** — Trajectory physics: boundary classes, walk strategies, coordinate conversion, dataset caching
- **`temporal/`** — Bit-stream datasets for density and parity tasks; KQ/GR evaluation metrics
- **`utils/`** — `BaseDataset`, parameter base classes, batch-continuity utilities

### `project/` — Task-Specific Code
| Sub-project | Purpose |
|---|---|
| `boolean_reservoir/` | Core `BooleanReservoir` model, LUT generation, binary encoding, graph construction |
| `path_integration/` | Train reservoir to predict final position from a continuous walk trajectory |
| `temporal/` | Train reservoir on temporal bit-stream tasks (density, parity) |
| `parallel_grid_search/` | Multi-process/multi-device parallel hyperparameter sweep |

---

## Sub-Projects

### Path Integration
The reservoir receives a sequence of displacement steps and must predict the agent's final position. Uses continuous trajectory data with a **temporal continuity** mechanism (`reset=False`): the reservoir state persists across batches so that successive steps are processed as one unbroken stream.

Walk strategies: `PhysicsWalkStrategy` (kinematics + friction), `SimpleRandomWalkStrategy`, `DiscreteRandomWalkStrategy`, `LevyFlightStrategy`.
Boundary types: `NoBoundary`, `IntervalBoundary` (1D), `PolygonBoundary` (2D shapely), `SoftBoundary` (smooth force field).

### Temporal Tasks
Binary bit-stream tasks where the reservoir must classify a sliding window:
- **Density**: does the window contain more 1s than 0s?
- **Parity**: does the window contain an odd number of 1s?

Evaluation metrics — **KQ** (Kernel Quality, tests input diversity) and **GR** (Generalization Rank, tests with partially identical inputs) — measure the reservoir's computational capacity.

### Parallel Grid Search
YAML configs support list-valued fields; `generate_param_combinations()` expands them into a cartesian-product grid. The `parallel_grid_search` sub-project runs each combination as an independent job, distributing work across CPUs and GPUs via a `ComputeJobResourceManager` and file-based locking. Results are aggregated into HDF5 logs.

---

## Configuration

All experiments are configured with YAML files that map 1:1 to Pydantic `Params` models. Any field can be a list to trigger a grid search:

```yaml
model:
  reservoir_layer:
    n_nodes: 1024
    k_avg: 3.0
    self_loops: [0, 0.1, 0.3]   # → 3 separate configs

dataset:
  train:
    encoding: ['cartesian', 'polar']  # → 2 separate configs
    boundary:
      params:
        rotation: pi/4          # math expressions are evaluated
```

Production configs live in `code/config/`. Test configs are co-located with each test suite under `project/<task>/test/config/`.

### Reproducibility

All random structures generated at initialization time — the reservoir graph, the LUT, and the initial node states — are produced via NumPy (not PyTorch), so results are **identical across CPU and GPU** given the same seed. The hot path (forward pass, reservoir ticks) uses pre-registered device buffers and is unaffected by this choice.

Seed control is via `set_seed(seed)` in `utils/utils.py`, called once per component before generation.

---

## Data

Data is generated on demand and cached under `data/`. The path is determined by a hash of the dataset parameters, so changing any parameter automatically regenerates data.

```
data/
├── path_integration/{coordinate}/{dimensions}D/{steps}s/{strategy}/{boundary}/
└── temporal/{task}/{dimensions}D/s-{sampling_mode}/b-{bits}/w-{window}/d-{delay}/
```

---

## Installation

### Option A — DevContainers (recommended)

Requires [VS Code](https://code.visualstudio.com/) and [Docker](https://docs.docker.com/get-started/get-docker/).

1. **Build the Docker image** (from the `docker/` directory):
   ```bash
   cd docker
   . ./setup.sh
   ```
   > Use `sudo` if your user lacks passwordless Docker access.

2. **Open in VS Code** — when prompted, choose *Reopen in Container*. The devcontainer mounts the repo read-only at `/repo` and the writable `code/` subtree at `/code`.

3. **Verify the environment** — a new terminal should show `(boolean_reservoir)` in the prompt. If not, rebuild the image (see [Troubleshooting](#troubleshooting)).

### Option B — Conda only

```bash
git clone --recursive https://github.com/chrisvibe/boolean_reservoir.git
cd boolean_reservoir
conda env create -f docker/src/environment.yaml
conda activate boolean_reservoir
export PYTHONPATH="$PWD/code"
```

Key dependencies: PyTorch 2.5.1 (CUDA 12.4), NumPy, SciPy, scikit-learn, NetworkX, Shapely, Pydantic 2, PyYAML, pytest.

---

## Running Tests

```bash
# Fast unit tests
pytest /code/project/boolean_reservoir/test/test_graphs.py -v
pytest /code/benchmark/ -v

# Path integration — continuity invariant check (fast)
pytest /code/project/path_integration/test/test_reset_false.py::test_reset_false_dataset_continuity -v

# Path integration — full suite (includes a short training run)
pytest /code/project/path_integration/test/ -v

# Temporal tasks
pytest /code/benchmark/temporal/test/test_temporal_dataset.py -v
```

---

## Troubleshooting
try deleting cached data if datasets changed and path didnt?

**GPU not detected**
The devcontainer declares `"gpu": "optional"` in `hostRequirements`. If no GPU is available, all training falls back to CPU automatically.

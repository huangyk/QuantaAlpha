# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

QuantaAlpha is an LLM-driven, self-evolving Alpha factor-mining framework built on top of **RD-Agent** (`rdagent==0.8.0`, pinned in `requirements.txt`). It takes a research direction as input, generates market hypotheses, turns them into Qlib factor expressions, backtests them, evolves the results through mutation/crossover, and stores validated factors in a factor library. A React + FastAPI web dashboard drives the same backend.

## Environment & Setup

Requires **Python 3.10+** (the README assumes a conda env named `quantaalpha`) and **Node 18+** for the frontend.

```bash
# 1. Create env and install the package in editable mode
conda create -n quantaalpha python=3.10
conda activate quantaalpha
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .   # defines the `quantaalpha` console script
pip install -r requirements.txt

# 2. Configure secrets + data paths (REQUIRED before any run)
cp configs/.env.example .env
# Edit .env: QLIB_DATA_DIR, DATA_RESULTS_DIR, OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, REASONING_MODEL
```

Without a valid `.env` (loader exits hard if missing) and a real `QLIB_DATA_DIR`, every entry point fails immediately. Data is downloaded from HuggingFace (`QuantaAlpha/qlib_csi300`): `cn_data.zip` (Qlib行情, must contain `calendars/ features/ instruments/`) and `daily_pv.h5` / `daily_pv_debug.h5` (precomputed price-volume HDF5 used during factor mining). The `run.sh` script symlinks `QLIB_DATA_DIR` into `~/.qlib/qlib_data/cn_data`.

## Common Commands

### Factor mining (the main experiment)

```bash
# Recommended: run.sh wraps env loading, experiment isolation, Qlib validation, symlink
./run.sh "price-volume factor mining"            # default experiment.yaml
./run.sh "momentum reversal" "exp_momentum"       # custom factor-library suffix

# Equivalent direct invocations (require `pip install -e .` + .env already loaded)
quantaalpha mine --direction "price-volume factor mining" --config_path configs/experiment.yaml
python launcher.py mine --direction "price-volume factor mining"
```

`mine` runs `AlphaAgentLoop` for `step_n` steps (default 100 = 20 loops × 5 steps). Loop resumption: `quantaalpha mine --path <session_folder>`. `STEP_N`, `CONFIG_PATH`, `EXPERIMENT_ID`, `FACTOR_LIBRARY_SUFFIX`, `USE_LOCAL` are honored as env vars by `run.sh`.

### Independent backtest (post-mining, V2)

There are **two distinct backtest entry points — do not confuse them**:

- `python -m quantaalpha.backtest.run_backtest` — standalone full backtest (Qlib LGB training + `TopkDropoutStrategy`). This is the documented "独立回测" path.
- `quantaalpha backtest` — runs `BacktestLoop` (an RD-Agent evolving loop over a single factor file), a different code path.

```bash
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source alpha158_20
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source custom \
    --factor-json all_factors_library.json
# combined = official baseline + your custom factors:
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source combined \
    --factor-json all_factors_library.json
# load factors only, no backtest:
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source alpha158_20 --dry-run -v
```

### Web dashboard

```bash
cd frontend-v2 && bash start.sh          # starts FastAPI backend (:8000) + Vite frontend (:3000)
# Manual split:
cd frontend-v2 && python backend/app.py   # backend (Terminal 1)
cd frontend-v2 && npm install && npm run dev   # frontend (Terminal 2)
```

### Lint / type-check (Python)

`requirements/lint.txt` lists `black, isort, mypy, ruff, toml-sort` (+ ruff stubs). **Note: the repo currently ships no ruff/black/isort/mypy config file**, so these tools run with their defaults if invoked (`ruff check quantaalpha/`, `black quantaalpha/`). Add a config block to `pyproject.toml` if you want enforced style.

### Tests

`requirements/test.txt` declares `pytest` + `coverage`, but **there is no test suite in the repo yet** (no `tests/` dir, no `test_*.py` files). When tests are added, the convention is:

```bash
pytest                                     # run all
pytest quantaalpha/path/to/test_module.py  # single file
pytest quantaalpha/path/to/test_module.py::test_fn   # single test
```

### Frontend build / lint

```bash
cd frontend-v2 && npm run build   # tsc + vite build
cd frontend-v2 && npm run lint    # eslint, --max-warnings 0
```

## High-Level Architecture

### The RD-Agent core abstractions (`quantaalpha/core/`)

The framework is a thin, finance-specific layer over RD-Agent's generic "hypothesis → experiment → feedback" loop:

- `Scenario` (`core/scenario.py`) — describes the task domain to the LLM (background, interface, output format, simulator, data intro). Factor scenario lives in `factors/experiment.py` (`QlibAlphaAgentScenario`).
- `Experiment` / `Task` / `Workspace` (`core/experiment.py`) — a factor experiment is a set of `FactorTask`s executed in a file-based workspace (`QlibFBWorkspace`).
- `Hypothesis` / `HypothesisFeedback` / `Trace` (`core/proposal.py`) — the knowledge trail; `Trace.hist` holds `(hypothesis, experiment, feedback)` tuples used for in-context learning.
- `Developer` (`core/developer.py`) — abstract `develop()`; realized as a **Coder** (generates/parses factor code) and a **Runner** (executes the backtest).
- The three "proposal" roles (`HypothesisGen`, `Hypothesis2Experiment`, `HypothesisExperiment2Feedback`) are the LLM call sites that drive propose → construct → feedback.

### The workflow engine (`quantaalpha/utils/workflow.py`)

`LoopBase` + the `LoopMeta` metaclass are the heart of execution. `LoopMeta` **auto-discovers workflow steps**: it collects every *public, callable* method name across the class MRO into a `steps` list. So a "loop" is just a class whose public methods become ordered steps, run repeatedly until `step_n` is exhausted or an error is raised. Each step receives the accumulated `loop_prev_out` dict and writes its result back into it. Sessions are pickled after every step to `__session__/<loop>/<step>_<name>` for resumption via `LoopBase.load`. **Adding a public method to a Loop subclass automatically makes it a workflow step** — keep helper methods prefixed with `_` so they are not picked up as steps.

### Component wiring by string class path (`quantaalpha/pipeline/settings.py`)

This is the **central extension point**. `AlphaAgentFactorBasePropSetting` (and siblings) hold dotted class-path strings for `scen`, `hypothesis_gen`, `hypothesis2experiment`, `coder`, `runner`, `summarizer`. At runtime `import_class()` instantiates them. To change mining behavior, **edit the class-path string in `settings.py`** rather than subclassing inline. Singletons: `ALPHA_AGENT_FACTOR_PROP_SETTING` (main mining), `FACTOR_PROP_SETTING`, `FACTOR_BACK_TEST_PROP_SETTING`, `FACTOR_FROM_REPORT_PROP_SETTING`, `MODEL_PROP_SETTING`.

### The mining loop (`quantaalpha/pipeline/loop.py`)

`AlphaAgentLoop` wires the proposal roles above into 5 steps: `factor_propose → factor_construct → factor_calculate → factor_backtest → feedback`. On each `feedback`, valid factors are auto-saved to `data/factorlib/all_factors_library[_<FACTOR_LIBRARY_SUFFIX>].json` via `factors/library.py:FactorLibraryManager`. `BacktestLoop` is the lighter variant used by `quantaalpha backtest`.

### Evolution (`quantaalpha/pipeline/evolution/`)

`EvolutionController` drives `Original → Mutation → Crossover → …` rounds. `StrategyTrajectory` + `TrajectoryPool` persist each trajectory's hypothesis/experiment/feedback and its primary metric (RankIC) to `trajectory_pool.json`. `MutationOperator` / `CrossoverOperator` spawn new directions from parent trajectories (parent selection strategy is configurable: `best` / `random` / `weighted` / `top_percent_plus_random`). Per-phase parallelism uses `multiprocessing`.

### Factors domain (`quantaalpha/factors/`)

Factors are **QLib expression strings** (Alpha158-style), not arbitrary Python. `qlib_coder.py:QlibFactorParser` parses them via `coder/expr_parser.py` + `coder/function_lib.py` (AST-based). `runner.py:QlibFactorRunner` runs the Qlib backtest locally (`USE_LOCAL=true`, default) or in a Docker container. `regulator/` enforces the **quality gate** (consistency / complexity / redundancy checks) configured under `quality_gate` in `experiment.yaml`. `loader/` ingests factors from JSON/PDF research material.

### Independent backtest (`quantaalpha/backtest/`, V2)

`run_backtest.py` → `runner.py:BacktestRunner` trains an LGB model on the factor set and runs a `TopkDropoutStrategy` backtest using Qlib, writing metrics to `data/results/backtest_v2_results`. Supports official sets (`alpha158` / `alpha158_20` / `alpha360`) and custom factor JSON from the library. Custom factors are computed with `factor_calculator.py` / `custom_factor_calculator.py` (expr parser + function lib). All parameters live in `configs/backtest.yaml`.

### LLM, parallelism, app, contrib

- `quantaalpha/llm/` — `LLMSettings` (pydantic-settings, env-driven) and `APIBackend` wrapping any OpenAI-compatible chat endpoint (`CHAT_MODEL` / `REASONING_MODEL`).
- `quantaalpha/coder/costeer/` — the CoSTEER evolving-code strategy from RD-Agent used to iteratively improve generated factor code.
- `quantaalpha/app/` — FastAPI backend (`backend/app.py`) for the web dashboard plus `health_check`, `collect_info`, and `benchmark/` (factor/model evaluation).
- `quantaalpha/contrib/model/` — optional graph-model experiment extension; not part of the default mining path.

## Configuration Map

| File | Purpose |
| :--- | :--- |
| `configs/.env.example` → `.env` | Secrets + data paths (`QLIB_DATA_DIR`, `DATA_RESULTS_DIR`, `OPENAI_*`, `CHAT_MODEL`, `REASONING_MODEL`). Loaded by every entry point. |
| `configs/experiment.yaml` | Mining: `planning`, `execution`, `evolution`, `quality_gate`, `factor`, `backtest`, `llm`, `logging`. |
| `configs/backtest.yaml` | Standalone V2 backtest: factor source, data window, dataset processors, model, strategy. |
| `quantaalpha/pipeline/settings.py` | Component class-path wiring (the main extension surface). |
| `quantaalpha/core/conf.py` | `RDAgentSettings`: `workspace_path`, `pickle_cache_folder_path_str`, file-lock behavior. |

## Operational Notes & Gotchas

- **No `.env` / bad `QLIB_DATA_DIR` ⇒ immediate hard exit.** Validate these first when a command fails before importing anything.
- **Windows timeouts** in `factor_mining.py` use a daemon thread + `os._exit(1)` (SIGALRM is Unix-only) — a forced timeout kills the process without cleanup; relevant when debugging long runs.
- **Session isolation:** `run.sh` creates per-experiment `WORKSPACE_PATH` and `PICKLE_CACHE` under `DATA_RESULTS_DIR` and stamps them with `EXPERIMENT_ID` (used to tag factors in the library).
- **Parallelism flags** are independent and easily multiplied: `planning.enabled` (N directions) + `execution.parallel_execution` (multiprocessing branches) + `evolution.parallel_enabled` (multiprocessing tasks). Watch CPU/memory.
- **Factor library file name** is customized via `FACTOR_LIBRARY_SUFFIX`; the default is `all_factors_library.json`.
- The README / `docs/PROJECT_STRUCTURE.md` predate the `evolution/`, `coder/`, `contrib/`, and `app/` modules — treat the actual `quantaalpha/` tree as the source of truth, not those docs.

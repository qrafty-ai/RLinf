# Tests Knowledge Base

**Generated:** 2026-02-16 05:07:36 UTC
**Commit:** 3d13b38
**Branch:** dev

## OVERVIEW
Pytest drives the suite; unit tests verify scheduler/worker primitives, and the e2e configs spin up Hydra+Ray stacks that mimic embodied, agentic, coding, and reasoning workloads.

## STRUCTURE
```
tests/
├── unit_tests/      # scheduler, worker, channel, auto placement, data utilities, comm regressions
└── e2e_tests/       # embodied/agent/reasoning/coding_online_rl/auto_placement/dynamic_scheduler configs + training_backend overrides
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a worker/scheduler regression | `tests/unit_tests/` | Large files like `test_comm.py`, `test_placement.py`, `test_worker.py` exercise Ray communication primitives; use fixtures that build fake clusters. |
| Register a new e2e config | `tests/e2e_tests/<domain>/` | Each directory mirrors an example (e.g., `embodied/`, `reasoning/`, `agent/`, `coding_online_rl/`). Hydra defaults compose with the same config library as `examples/<domain>/config`. |
| Extend CI targets | `.github/workflows/ci-tests.yml` + `tests/unit_tests/` | The workflow loops over filters and requires `run-ci` label; new tests should be referenced there if they require special hardware. |

## CONVENTIONS
- Tests prefer class-based fixtures with descriptive docstrings; assert statements are explicit, and `pytest.raises` is used for error cases. Large fixtures often reuse `Cluster`, `WorkerGroup`, and `ComponentPlacement` helpers from `rlinf/scheduler`.\
- Many unit tests accept both CPU and GPU by guarding with `torch.cuda.is_available()`; they avoid hard assumptions about `cuda` availability.\
- e2e jobs are Hydra-driven; run them through the matching `examples/` script or the `tests/e2e_tests/<domain>/<config>.yaml` landing page (they assume the same `source .envrc`/uv environment as the examples).\
- Keep `tests/` under `pyproject.toml` linting (Ruff + isort) and do not touch generated `_build`/`.pixi` artifacts inside tests.

## ANTI-PATTERNS
- Do not commit tests that rely on untracked datasets; e2e configs assume `portal` datasets (e.g., ManiSkill, IsaacLab) are prepopulated under `/workspace/dataset`. Document missing data in the config docstring instead.\
- Do not remove the `run-ci` label gating or bypass the `ci-tests` coverage check; the filter coverage job enforces that every new file is covered by a workflow filter.\
- Avoid `sleep`-heavy `ray.get()` loops that stall the test harness; rely on `WorkerGroup.wait()` helpers instead.

## COMMANDS
```
source .envrc
uv venv && uv sync && uv pip install -r pixi.lock
pytest tests/unit_tests/
UV_TORCH_BACKEND=auto pytest tests/unit_tests/test_worker.py::TestWorker
```

## NOTES
- Unit tests live in `tests/unit_tests/` with the same import path as the package under `rlinf/`; keep the `tests/unit_tests` directory tree in sync with the namespace.\
- e2e tests reuse the Hydra config defaults under `examples/<domain>/config`; prefer adding a new Hydra override instead of writing bespoke scripts.\

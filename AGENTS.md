# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-16 05:07:36 UTC
**Commit:** 3d13b38
**Branch:** dev

## OVERVIEW
RLinf is a Ray‑backed reinforcement-learning infrastructure that wires Hydra configs, Ray actors, and large models into embodied/agentic training loops; the codebase bifurcates into a core `rlinf/` package, Hydra entry scripts under `examples/`, and the supporting docs/tests/tooling around them. Before touching any scripts, always `source .envrc` so the expected environment variables, paths, and uv settings are loaded (the repo relies on that hook for GPU selection, cache locations, and `PYTHONPATH`).

## STRUCTURE
```
./
├── rlinf/          # core package: algorithms, scheduler, workers, env wrappers, models, agents
├── examples/        # Hydra entry scripts + config libraries for embodied, reasoning, agentic, coding, multiturn demos
├── docs/            # bilingual (EN/ZH) Sphinx docs with `autobuild.sh`
├── tests/           # unit_tests/ (workers, scheduler, utilities) and e2e_tests/ (embodied/agent/reasoning)
├── requirements/    # install.sh and pack lists per profile
├── .github/         # self-hosted workflows, label gating, install/dock builds
├── docker/          # Dockerfiles/targets used by CI and release bundles
├── toolkits/        # helper scripts (replay buffer, eval, auto placement) that share utils with core loops
├── ray_utils/       # helper Ray cluster/start scripts reused in scripts and docs
└── pixi.{lock,toml}  # uv lockfile + manifest for uv-based dependency management
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add/extend algorithms or reward/loss registries | `rlinf/algorithms/` | Register via `register_advantage`, `register_loss`, `register_reward` to keep Hydra/defaults in sync.
| Wire a new env/model/worker + scheduler change | `rlinf/{envs,models,workers,scheduler}/` | Config lives once, entry scripts instantiate via `Worker.create_group().launch()` and rely on `rlinf/config.py`. 
| Launch or extend examples | `examples/<domain>/` | Each folder owns Hydra defaults under `config/`, helper `run_*.sh`, and scripts decorated with `@hydra.main`. 
| Document features, tutorials, or CLI options | `docs/` + `docs/source-*/` | Use the Sphinx `Makefile`/`autobuild.sh` routine; keep translations aligned. 
| Update tests or add e2e coverage | `tests/unit_tests/` + `tests/e2e_tests/` | Unit tests use PyTest fixtures; e2e tests mirror config combos (embodied, reasoning, agent, coding, auto placement, dynamic scheduler). 
| Adjust CI/install flows | `.github/workflows/`, `requirements/`, `pixi.*` | Workflows are `workflow_call` driven, self-hosted, and call `uv` commands plus `bash requirements/install.sh`; the `ci-tests` job enforces the `run-ci` label and draft rejection.

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `rlinf.config.build_config` | function | `rlinf/config.py` | ~1.2k lines | composes Hydra defaults for models, envs, algorithm/backbone settings. |
| `rlinf.scheduler.Cluster` | class | `rlinf/scheduler/cluster/cluster.py` | core Ray cluster + hardware placement with placement strategies. |
| `examples/embodiment/train_embodied_agent.py` | script | `examples/embodiment/` | Hydra entry and the go-to example for single-node blueprint. |
| `tests/unit_tests/test_comm.py` | tests | `tests/unit_tests/` | gated by `DO NOT` for worker communication invariants; also the largest unit test file. |

## CONVENTIONS
- Hydra is the central config router: add new nodes in `rlinf/config.py` and reference them via `defaults` or `@` override aliases; avoid mutating the `cfg` object at import time or in Hydra defaults (per the CONTRIBUTING warning). Use `@hydra.main` entry scripts from `examples/` rather than calling core modules directly.
- Ray must be started manually (`ray start --head` or via `ray_utils/start_ray.sh`); workers are launched through `Worker.create_group().launch(...)` and expect `cluster.num_nodes`, `component_placement`, and `node_groups` to align with the Ray actors created in `rlinf/scheduler`.
- Dependency management leans on `uv`. Always `source .envrc`, `uv venv`, `uv sync`, then `uv pip install ...` before running binaries; this also ensures the uv lockfiles (`pixi.lock`, `pixi.toml`) stay in sync with `requirements/install.sh`.
- CI runs on self-hosted runners (`runs-on: self-hosted`/`embodied`/`reason`) and is gated by the `run-ci` label. Draft PRs are blocked, and the workflows deliberately sync `uv` caches and remove node_modules/`uv` caches in their "maximize storage" steps.
- Tests and examples rely on Hydra overrides (e.g., `@env.train`, `@actor.model`) and do not expect dynamic config patches outside the YAML/DictConfig system.
## ANTI-PATTERNS (THIS PROJECT)
- `DO NOT` perform calculations or set dynamic values inside YAMLs—the CONTRIBUTING guide forbids writing dynamic values in configs; add config options instead.
- `DO NOT` run `ci-tests` on a PR without the `run-ci` label or on a draft, as the workflows explicitly guard both conditions.
- `DO NOT` edit generated artifacts under `docs/_build/` or commit files that are produced at runtime (`.pixi/`, `.envrc` entries, `_build`).
## UNIQUE STYLES
- Config files use the `<domain>/<name>@<scope>` aliasing pattern (`env/<scenario>@env.train`) to compose Hydra runs; new defaults must be added to `rlinf/config.py` so they appear in that schema.
- The repo centers on a three-way combo: Ray workers (Actor/Rollout/Reward), Hydra configs, and `uv` for dependency + script isolation—expect `uv` commands sprinkled across workflows and docs.
- Entry scripts are grouped under `examples/` with supporting `run_*.sh` wrappers to keep Ray envs and `hydra.run.dir` pinned to repository subdirs.
- `AGENTS.md` is the living knowledge base; use these files before calling helper scripts, and update them when you introduce new domains (tests, docs, examples, etc.).
## COMMANDS
```
source .envrc && uv venv && uv sync                          # bootstrap uv-managed env
uv pip install --upgrade ruff && uv pip install -r pixi.lock  # keep lint + deps aligned
bash requirements/install.sh embodied --model <model> --env <env>
python examples/embodiment/train_embodied_agent.py --config-name <config>
pytest tests/unit_tests/
bash docs/autobuild.sh                                         # live-preview Sphinx docs (EN+ZH)
```

## NOTES
- CI uses the `run-ci` label and is self-hosted, so check `.github/workflows/ci-tests.yml` before modifying workflow logic.
- Tests/e2e configs are Hydra-driven; add new `tests/e2e_tests/<domain>/*.yaml` plus matching `examples/<domain>/config/` files when extending behavior.
- `source .envrc` is mandatory before `uv` commands; the hook sets `PYTHONPATH`, `RLINF_NODE_RANK`, and data directories referenced elsewhere.

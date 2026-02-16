# Examples Knowledge Base

**Generated:** 2026-02-16 05:07:36 UTC
**Commit:** 3d13b38
**Branch:** dev

## OVERVIEW
Every training job or evaluation script lives under `examples/`; each subdirectory owns a Hydra config library in `config/`, a Python entry script decorated with `@hydra.main`, and shell helpers that keep `hydra.run.dir` and Ray placement pinned to repo subdirs.

## STRUCTURE
```
examples/
├── embodiment/       # VLA/simulator training (train_embodied_agent, train_async, eval, collect_real_data)
├── reasoning/        # GRPO/math proofing entry scripts
├── coding_online_rl/ # online coding RL client + judge loops
├── multiturn_demo/   # tool-enabled agentic Playwright/demo harnesses
├── searchr1/         # index builder + evaluation server + main agent launcher
├── sft/              # supervised fine-tuning wrappers
└── wideseek_r1/      # documentation-only placeholder
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a training Hydra config | `examples/<domain>/config/` | Use the `<component>@<scope>` aliasing pattern (`model/openvla@actor.model`, `env/maniskill@env.train`). Keep `hydra.run.dir` rooted under the example directory so checkpoints land in `outputs/`. |
| Launch a demo loop | `examples/<domain>/<script>.py` | Each script calls `build_config`, `Cluster.create_group().launch(...)`, and may invoke `run_async` or `eval`. Use the companion `run_*.sh` wrappers when running locally. |
| Extend SearchR1 | `examples/searchr1/` + `rlinf/agents/searchr1/` | `main_searchr1.py` powers the planner; `index_builder.py` precomputes retrieval data and must finish before evaluation. |
| Build a coding or multiturn agent | `examples/coding_online_rl/`, `examples/multiturn_demo/` | These scripts spin up LLM judges and stateful tool loops; they rely on the `examples/<domain>/config/` directory for Hydra overrides. |

## CONVENTIONS
- Scripts source `run_*.sh` wrappers that set `source .envrc`, `uv venv`, and the `PYTHONPATH`/`RLINF_NODE_RANK` guard rails before invoking the Python module with `--config-name`. Avoid copy/pasting Hydra overrides outside these config libraries.
- Config defaults are versioned via `@`-scoped files; add new options by dropping files into the domain’s `config/` tree and referencing them via `defaults` lists. Immediate overrides (like `${oc.env:...}`) stay in YAML, not in Python.
- `examples/embodiment` uses ManiSkill/IsaacLab real robot wrappers, so follow the `env/` config naming conventions (e.g., `env/maniskill_put_carrot_on_plate_in_scene@env.train`). `examples/searchr1` uses retrieval indexes and must call `index_builder.py` before launching `main_searchr1.py`. `coding_online_rl` and `multiturn_demo` are agentic loops that rely on `rlinf/agents` helpers and the `mcp` tool. 

## ANTI-PATTERNS
- Don’t bake environment-specific secrets into example configs; keep credential placeholders in Hydra defaults and override via `hydra.run.dir` or environment variables.
- Avoid running the Python entry script directly without the `run_*.sh` wrappers in CI—they enforce the expected `uv`/Ray environment (especially `RLINF_NODE_RANK`).

## COMMANDS
```
source .envrc
uv venv && uv sync
bash examples/embodiment/run_embodiment.sh <config_name>
bash examples/searchr1/run_main_searchr1_single.sh <config_name>
python examples/coding_online_rl/main_coding_online_rl.py --config-name coding_online_rl/online_rl
bash examples/multiturn_demo/run_main_tool.sh <config>
```

## NOTES
- `examples/wideseek_r1/` only contains documentation; there is currently no runnable entrypoint there.\
- The entry scripts feed data into `toolkits/` helpers (replay buffer converters, eval scripts) when exporting checkpoints or building assets.

# RLinf Core Package

**Generated:** 2026-02-16 05:07:36 UTC
**Commit:** 3d13b38
**Branch:** dev

## OVERVIEW
The `rlinf/` package contains the distributed pieces that power the training loops—algorithms, schedulers, env wrappers, workers, and models. Most entry paths live in `examples/`, so this package is built to be imported and instantiated, not executed directly.

## STRUCTURE
```
rlinf/
├── agents/         # agent loops (SearchR1, multiturn demo, tool agent helpers)
├── algorithms/     # advantage/loss/reward registries + implementations
├── config.py       # Hydra config builder + validation (SupportedModel, SupportedEnvType)
├── envs/           # ManiSkill/Behavior/IsaacLab/realworld wrappers + `get_env_cls`
├── hybrid_engines/  # SGLang/vLLM/FSDP/Megatron helpers for rollout + inference
├── models/          # embodied/action/action models (OpenVLA, OpenPI, world models)
├── runners/         # embodied/agent/reasoning runner loops
├── scheduler/       # Ray cluster, placement, hardware, collective/channel utilities
├── utils/           # distributed helpers, data iterators, checkpoint conversion
└── workers/          # actor/rollout/env/reward worker implementations + base Worker
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a policy/algorithm advantage | `rlinf/algorithms/` | Register via `register_advantage`, `register_policy_loss`, `register_reward`; `registry.py` drives dispatch. |
| Introduce a new environment | `rlinf/envs/` + `envs/__init__.py` | Add `SupportedEnvType` entry, return the class from `get_env_cls`, keep heavy dependencies lazy. |
| Harden distributed communication | `rlinf/scheduler/collective/`, `rlinf/utils/distributed.py` | Shared queues, tensor dataclasses, and Channel abstractions live here; `collective_group.py` is the hottest file. |
| Extend worker behavior | `rlinf/workers/` + `rlinf/scheduler/placement/` | All workers subclass `Worker` and call `WorkerGroup.create(...).launch(...)`; placement strategies live under `scheduler/placement`. |
| Wire new model/config | `rlinf/models/embodiment/` + `rlinf/config.py` | Models register via `build_config`/`get_model_config_and_processor`; use factory helpers in `models/__init__.py`. |

## CONVENTIONS
- Hydra configuration is centralized in `rlinf/config.py`. The entry scripts pass `cfg` down, so avoid runtime imports that mutate `cfg`; add new config keys and defaults there before touching downstream modules.
- Workers, runners, and schedulers expect Ray to be running; `Worker.create_group().launch()` is the canonical way to start actors and rollout workers, so keep placement, hardware ranks, and component mapping consistent.
- Models and envs rely on registries (`get_supported_model`, `register_reward`, `get_env_cls`). Keep heavy frameworks inside the specific subpackage to keep imports lightweight (e.g., `OpenVLA` under `models/embodiment/openvla`).
- Data movement uses tokio-style channels (`rlinf/scheduler/channel/`) plus packed placers. The `collective_group` class and related `NodeGroupInfo` structures are the most touched files when debugging placement/communication.

## ANTI-PATTERNS
- Avoid instantiating Ray actors or heavy models at import time—instantiate inside Hydra entry points so Ray serialization and uv paths stay predictable.
- Do not duplicate placement mapping logic; new strategies should extend the existing `PlacementStrategy` hierarchy to avoid drift.

## UNIQUE STYLES
- The `rlinf/agents/` subpackage contains helper loops for SearchR1 and multiturn demos; look there for tool invocation + LM orchestration patterns.
- The `hyper` directories (e.g., `hybrid_engines/vllm`, `utils/ckpt_convertor/fsdp_convertor`) implement converter scripts that are reused by toolkits and doc demo scripts.
- Models expose `BasePolicy`/`ActionModel` lifecycles: `default_forward`, `predict_action_batch`, loss-specific entry points for FSDP vs Megatron.

## COMMANDS
There is no standalone CLI inside `rlinf/`; drive everything through the Hydra entry scripts under `examples/` so the package stays import-only.

## NOTES
- Changes to `rlinf/config.py` ripple through every entry script—update the `build_config` defaults before adding new model/env knobs.
- `examples/` and `tests/` are the best smoke tests for any change inside `rlinf/`; they re-assemble configs, launch Ray, and exercise Ray actors in a repo snapshot.

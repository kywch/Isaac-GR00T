# AGENTS.md — Isaac-GR00T

## What This Repo Is

Isaac-GR00T is a framework for training and deploying Vision-Language-Action (VLA) models for multi-embodiment robotic control. The core model (GR00T-N1D6) uses high-resolution vision and language instructions to generate action trajectories via a Diffusion Transformer head.

## Exploring the Codebase

This repo has a **wiki** (`wiki/`) that serves as a semantic guide to the code. When exploring, always use the wiki and the source together — the wiki explains *how* things connect; the code shows *what* exactly happens.

### Start here

1. Read `wiki/overview.md` first. It maps the entire system: data ingestion → training → inference → deployment.
2. Use `wiki/module_tree.json` to look up which classes/functions belong to which module and where they live in the source tree.
3. Use `wiki/dependency_graph.json` for call relationships, file paths, and line numbers when you need to trace how components connect.

### Module guide

Each wiki page covers one functional module. Read the relevant wiki page **before** diving into its source directory:

| Wiki Page | Source Directory | What It Covers |
|---|---|---|
| `wiki/configuration.md` | `gr00t/configs/` | Type-safe config system (model hyperparams, dataset specs, training settings) |
| `wiki/data_pipeline.md` | `gr00t/data/` | Loading, processing, batching multi-modal robotic data |
| `wiki/model_architecture.md` | `gr00t/model/` | VLA neural networks — top-level organization |
| `wiki/gr00t_n1d6_model.md` | `gr00t/model/gr00t_n1d6/` | The main GR00T-N1D6 model and action head |
| `wiki/eagle_backbone.md` | `gr00t/model/eagle/` | Visual backbone (Eagle vision encoder) |
| `wiki/nvidia_eagle_vl.md` | `gr00t/model/eagle/` | NVIDIA Eagle vision-language model details |
| `wiki/diffusion_transformer.md` | `gr00t/model/modules/dit.py` | DiT blocks for action generation |
| `wiki/flowmatching_modules.md` | `gr00t/model/modules/` | Flow matching / diffusion sampling |
| `wiki/embodiment_conditioned_mlp.md` | `gr00t/model/modules/` | Embodiment-conditioned layers |
| `wiki/base_pipeline.md` | `gr00t/model/base/` | Base model pipeline abstraction |
| `wiki/training_engine.md` | `gr00t/experiment/` | Training framework, metrics, optimization |
| `wiki/policy_inference.md` | `gr00t/policy/` | Running models: local, networked, replay modes |
| `wiki/evaluation_environments.md` | `gr00t/eval/` | Gym wrappers, sim environments, rollout recording |
| `wiki/deployment.md` | `scripts/`, `gr00t/` | ONNX export, TensorRT acceleration |

### How to explore a topic

When asked about any part of this codebase:

1. **Orient via wiki** — Read the matching wiki page to understand the module's purpose, key classes, and how it fits into the larger system.
2. **Locate source** — Use the module table above or `wiki/module_tree.json` to find the exact source files.
3. **Read the code** — Read the actual implementation for precise details.
4. **Trace connections** — Use `wiki/dependency_graph.json` or grep for imports/call sites to understand how the component interacts with others.

### Using subagents for exploration

For broad exploration tasks, spawn Explore subagents scoped to specific modules. Each subagent should:
- Read the relevant wiki page first for context
- Then explore the corresponding source directory
- Report back with findings grounded in both the wiki explanation and the actual code

Example: to understand the model architecture, spawn one agent to read `wiki/model_architecture.md` + explore `gr00t/model/`, and another to read `wiki/data_pipeline.md` + explore `gr00t/data/` — in parallel.

### Other useful entry points

- `examples/` — Usage examples and demo scripts
- `getting_started/` — Setup and quickstart guides
- `scripts/` — Training, evaluation, and deployment scripts
- `pyproject.toml` — Dependencies and project metadata
- `demo_data/` — Sample data for testing

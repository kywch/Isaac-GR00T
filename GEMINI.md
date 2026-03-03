## Mandatory Rules
- **The wiki is your guide; the source code is the truth.** Follow this exact sequence for every question about this codebase:
  1. Check the **Wiki Reference** table below to identify which wiki page(s) cover the topic.
  2. Read the relevant `wiki/<module>.md` page(s) to understand architecture, key classes, and integration points.
  3. Use `wiki/module_tree.json` to locate specific classes/functions and their source paths.
  4. Use `wiki/dependency_graph.json` to trace call relationships, file paths, and line numbers between components.
  5. **Then open the source files** — guided by what the wiki told you — and **verify against the actual code**.
- Skipping steps 1–4 and jumping straight to `grep_search` / `view_file` on source code is **not allowed**.
- The wiki may not perfectly match the current code. Always use it to orient your exploration, then cross-check the actual source for ground truth.

## Wiki Reference

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

### Supplementary data files
- `wiki/module_tree.json` — maps classes/functions to modules and source paths
- `wiki/dependency_graph.json` — call relationships, file paths, and line numbers between components
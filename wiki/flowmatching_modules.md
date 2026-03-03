# Flow-Matching Modules

The **flowmatching_modules** module provides specialized neural network components designed for flow-matching and diffusion-based action prediction within the GR00T architecture. It focuses on encoding continuous action trajectories and temporal conditioning into high-dimensional latent representations.

---

## Overview

The module serves as a utility layer for generative modeling tasks, specifically those involving time-dependent probability paths. Its key responsibilities include:
- Transforming raw action vectors into latent embeddings using multi-layer perceptrons.
- Generating sinusoidal embeddings for continuous or discrete timesteps to provide temporal context.
- Fusing action and time information into a unified representation for downstream processing by models like the [`DiT`](../gr00t/model/modules/dit.py#L172).

---

## Architecture

The architecture is built around three core classes that operate on tensor sequences. The [`ActionEncoder`](../gr00t/model/modules/flowmatching_modules.py#L54) acts as the primary orchestrator, utilizing [`SinusoidalPositionalEncoding`](../gr00t/model/modules/flowmatching_modules.py#L10) for its temporal logic.

### Data Flow Diagram
```mermaid
graph TD
    A[Actions (B, T, D)] --> B[ActionEncoder]
    T[Timesteps (B,)] --> B
    
    subgraph ActionEncoder
        B --> C[Linear W1]
        T --> D[SinusoidalPositionalEncoding]
        C --> E[Concat]
        D --> E
        E --> F[Linear W2 + Swish]
        F --> G[Linear W3]
    end
    
    G --> H[Action Latents (B, T, H)]
```

The process begins by projecting raw actions into a hidden space while simultaneously generating temporal encodings. These two streams are concatenated and passed through a non-linear bottleneck defined by the [`swish`](../gr00t/model/modules/flowmatching_modules.py#L6) activation function.

---

## Core Components

> **Start here:** [`ActionEncoder`](../gr00t/model/modules/flowmatching_modules.py#L54) — read its [`forward()`](../gr00t/model/modules/flowmatching_modules.py#L65) method first to understand how actions and time are fused.

### `ActionEncoder`
The **ActionEncoder** is the central component for preparing action sequences for generative modeling. It ensures that every step in an action trajectory is aware of the current diffusion or flow-matching timestep.

- **Key Entry Point:** [`forward(actions, timesteps)`](../gr00t/model/modules/flowmatching_modules.py#L65)
  - **Does:** Encodes a trajectory of actions `(B, T, D)` conditioned on a batch of timesteps `(B,)`.
  - **Returns:** A tensor of embeddings `(B, T, hidden_size)`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `action_dim` | `int` | N/A | Dimensionality of the input action space. |
| `hidden_size` | `int` | N/A | Dimensionality of the resulting embeddings and internal layers. |

### `SinusoidalPositionalEncoding`
The **SinusoidalPositionalEncoding** class implements a standard harmonic encoding scheme. It maps scalar timesteps to a high-dimensional space where temporal proximity is reflected in the dot product of the embeddings.

- **Key Entry Point:** [`forward(timesteps)`](../gr00t/model/modules/flowmatching_modules.py#L19)
  - **Does:** Computes sine and cosine frequencies for a given set of timesteps.
  - **Returns:** An encoding tensor of shape `(B, T, embedding_dim)`.

### `SmallMLP`
The **SmallMLP** is a general-purpose utility for lightweight feature transformation.

- **Key Entry Point:** [`forward(x)`](../gr00t/model/modules/flowmatching_modules.py#L49)
  - **Does:** Passes input through two linear layers with a ReLU activation in between.
  - **Returns:** Transformed features of the requested `output_dim`.

---

## Usage & Extension

### Extension Points
- **Activation Functions:** The module currently uses a custom [`swish`](../gr00t/model/modules/flowmatching_modules.py#L6) function. Developers can replace this with `F.silu` or other gated activations if required.
- **Encoding Logic:** The [`ActionEncoder`](../gr00t/model/modules/flowmatching_modules.py#L54) can be extended to support additional conditioning (e.g., state-based conditioning) by modifying the concatenation step in [`forward`](../gr00t/model/modules/flowmatching_modules.py#L65).

### Error Handling
- The [`ActionEncoder`](../gr00t/model/modules/flowmatching_modules.py#L54) validates input shapes during [`forward`](../gr00t/model/modules/flowmatching_modules.py#L65). It raises a `ValueError` if the `timesteps` tensor is not a 1D tensor matching the batch size of the `actions`, as it needs to replicate the scalar time across the sequence length `T`.

---

## Integration

This module provides critical utilities for several high-level components:
- [**Diffusion Transformer**](diffusion_transformer.md): Uses these encoders to process noisy action samples and temporal indices.
- [**GR00T-N1D6 Model**](gr00t_n1d6_model.md): Employs the [`ActionEncoder`](../gr00t/model/modules/flowmatching_modules.py#L54) within its generative heads to produce final control outputs.
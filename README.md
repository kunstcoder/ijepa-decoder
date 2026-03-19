# I-JEPA Decoder Prototype for Semantic Hole Reconstruction

This repository implements a small downstream prototype for **semantic hole reconstruction / inpainting** using an **I-JEPA-style** encoder, predictor, target encoder, and image decoder pipeline.

The main idea is:

1. mask part of an image,
2. convert the mask into patch-aligned visible / hole indices,
3. encode visible context tokens,
4. predict semantic tokens for the missing patches,
5. scatter predicted hole tokens back into a full spatial token canvas,
6. decode that token canvas into RGB,
7. optimize with both semantic alignment loss and hole-only reconstruction loss.

This README focuses on the **implemented code**, **requirements**, and **how to run the available scripts**.

---

## What the Code Implements

### 1. Mask processing

Implemented in `data/mask_samplers.py`.

- `PatchMaskConverter`
  - converts a binary pixel mask `[B, 1, H, W]` into patch-level hole/visible indices,
  - produces:
    - `visible_indices`,
    - `hole_indices`,
    - `patch_mask`.
- `apply_pixel_mask`
  - zeros masked pixels before patchification,
  - helps reduce information leakage when a masked region only partially overlaps a patch.

### 2. I-JEPA wrapper and checkpoint loading

Implemented in `models/ijepa_wrapper.py`.

- `IJEPAWrapper`
  - wraps:
    - `encoder`,
    - `predictor`,
    - `target_encoder`.
- `from_checkpoint(...)`
  - loads component weights from nested or flat checkpoint dictionaries,
  - supports common prefixes such as `encoder`, `predictor`, `target_encoder`, `teacher_encoder`, and `module.encoder`.
- fallback module inference
  - if no architecture-specific factory is supplied, the loader can infer:
    - a single `LinearProjection`, or
    - a stacked `SequentialMLP`.
- training controls
  - `set_trainable(...)` toggles frozen/trainable modules,
  - `update_target_encoder(...)` applies EMA updates from encoder to target encoder.

### 3. Hole token scatter and spatial canvas rebuild

Implemented in `models/hole_token_scatter.py`.

- `scatter_hole_tokens(...)`
  - merges visible tokens and predicted hole tokens into a full `[B, N_patch, D]` canvas.
- `reshape_token_canvas(...)`
  - converts `[B, N_patch, D]` into `[B, D, H_patch, W_patch]` for decoding.

### 4. Baseline decoder

Implemented in `models/decoder_baseline.py`.

- `ConvDecoder`
  - a minimal convolutional decoder,
  - upsamples a token grid into an RGB image,
  - intended for quick end-to-end validation of the latent-to-image path.

### 5. Losses

Implemented in `models/losses.py`.

- `SemanticReconstructionLoss`
  - semantic term: Smooth L1 between predicted hole tokens and teacher hole tokens,
  - reconstruction term: masked L1 on the hole region only,
  - total: `lambda_jepa * semantic + lambda_rgb * reconstruction`.

### 6. Training stage configuration and dry-run training path

Implemented in `train/train_semantic_inpaint.py`.

- `TrainConfig`
  - holds model/training hyperparameters.
- `TRAINING_STAGES`
  - currently defines:
    - `decoder_warmup`,
    - `predictor_finetune`,
    - `end_to_end`.
- `configure_training_stage(...)`
  - applies freeze/unfreeze logic to wrapper modules.
- `build_optimizer(...)`
  - creates parameter groups for encoder / predictor / decoder.
- `run_dry_training_step(...)`
  - executes a synthetic training step using random tensors,
  - verifies mask conversion, stage setup, token scatter, decoder forward, loss computation, optimizer grouping, and optional EMA update.

### 7. Tests

Implemented in `tests/test_semantic_inpaint.py`.

Current tests cover:

- patch mask conversion,
- pixel masking,
- hole token scatter alignment,
- checkpoint extraction,
- trainable flag configuration,
- EMA update behavior,
- optimizer parameter grouping,
- dry-run training behavior for multiple stages.

---

## Repository Layout

```text
configs/
  downstream_semantic_inpaint.yaml
data/
  mask_samplers.py
models/
  decoder_baseline.py
  hole_token_scatter.py
  ijepa_wrapper.py
  losses.py
tests/
  test_semantic_inpaint.py
train/
  train_semantic_inpaint.py
IMPLEMENTATION_PLAN.md
README.md
```

---

## Requirements

The project currently depends on a minimal Python scientific / testing stack.

### Recommended environment

- Python 3.10+
- PyTorch
- pytest

### Suggested install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytest
```

If you plan to extend the project with YAML config loading or dataset code later, you may also want:

```bash
pip install pyyaml
```

> Note: this repository does not currently include a pinned `requirements.txt` or `pyproject.toml`, so dependencies are documented here rather than installed from a lockfile.

---

## Configuration

The example config file is:

```text
configs/downstream_semantic_inpaint.yaml
```

It currently documents the main knobs used by the prototype:

- image and patch sizing,
- token dimension,
- batch size,
- semantic / RGB loss weights,
- active training stage,
- masking behavior,
- EMA settings,
- optimizer learning rates.

Example fields:

```yaml
image_size: 32
patch_size: 8
token_dim: 16
batch_size: 2
lambda_jepa: 1.0
lambda_rgb: 0.5
stage: decoder_warmup
```

The Python-side configuration object is `TrainConfig` in `train/train_semantic_inpaint.py`.

---

## Training Stages

The implemented stage presets are:

### `decoder_warmup`

- encoder: frozen
- predictor: frozen
- target encoder: frozen
- decoder: trainable
- EMA update: disabled

### `predictor_finetune`

- encoder: frozen
- predictor: trainable
- target encoder: frozen
- decoder: trainable
- EMA update: enabled

### `end_to_end`

- encoder: trainable
- predictor: trainable
- target encoder: frozen
- decoder: trainable
- EMA update: enabled

These stage behaviors are defined in code, not just in YAML, through `TRAINING_STAGES` and `configure_training_stage(...)`.

---

## Available Scripts

## 1. Dry training script

Main entrypoint:

```bash
python train/train_semantic_inpaint.py
```

What it does:

- creates synthetic images and a simple binary mask,
- converts the mask to patch indices,
- runs the wrapper encoder / predictor / target encoder,
- rebuilds the token canvas,
- decodes into RGB,
- computes the combined loss,
- prints metrics for the selected stage.

This is currently the main executable training-related script in the repository.

### Example output fields

The script prints metrics such as:

- `stage_name`
- `semantic_loss`
- `reconstruction_loss`
- `total_loss`
- `encoder_trainable`
- `predictor_trainable`
- `optimizer_param_groups`
- per-group learning rates / parameter counts

## 2. Test script

Run the unit tests with:

```bash
python -m pytest
```

or:

```bash
pytest
```

---

## Code Flow Summary

The currently implemented tensor flow is:

```text
image
  -> binary mask
  -> apply_pixel_mask(...)
  -> PatchMaskConverter(...)
  -> visible_indices / hole_indices
  -> wrapper.encode_context(...)
  -> wrapper.predict_holes(...)
  -> wrapper.encode_target(...)
  -> scatter_hole_tokens(...)
  -> reshape_token_canvas(...)
  -> ConvDecoder(...)
  -> SemanticReconstructionLoss(...)
```

This path is fully exercised by `run_dry_training_step(...)`.

---

## Checkpoint Loading Usage

If you have a compatible checkpoint, the intended loading path is:

```python
from models.ijepa_wrapper import IJEPAWrapper

wrapper = IJEPAWrapper.from_checkpoint("path/to/checkpoint.pt")
```

If the checkpoint structure does not match the fallback inference logic, you can extend the loader by passing explicit `module_factories` for `encoder`, `predictor`, and `target_encoder`.

---

## Current Limitations

The current repository is a **prototype scaffold** around core tensor plumbing. In particular:

- there is no real dataset loader yet,
- there is no full training loop over a dataset,
- there is no evaluation or visualization script yet,
- irregular/free-form mask sampling is not implemented yet,
- checkpoint loading is generic and may need architecture-specific factories for real I-JEPA checkpoints.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytest
python train/train_semantic_inpaint.py
python -m pytest
```

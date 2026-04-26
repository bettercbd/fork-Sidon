# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Sidon is a two-stage speech restoration system: a LoRA-adapted w2v-BERT 2.0 **feature predictor** that denoises SSL representations, and a **vocoder** (decoder + discriminator) that resynthesises waveforms from those features. The repo ships training, preprocessing, and dataset-cleansing pipelines built on PyTorch Lightning + Hydra, with WebDataset as the on-disk format.

## Environment

- Python 3.10+, managed with `uv` (`uv sync` to install). `pyproject.toml` is the source of truth for deps.
- GPU/CUDA assumed for any training or inference run. Most config defaults (batch sizes, paths under `/groups/...`) target an internal PBS cluster — override them via Hydra for local runs.

## Common commands

```bash
uv sync                                          # install deps
uv run python -m compileall src                  # quick syntax sweep before submitting jobs

# Generate WebDataset shards from raw audio
uv run python -m sidon.preprocess \
  data=webdataset_preprocess_24k \
  preprocess.writer_name=my_run

# Training (three sequential stages — see "Training pipeline" in README.md)
uv run python -m sidon.train model=sidon_feature_predictor data=preprocessed
uv run python -m sidon.train model=sidon_vocoder_pretrain  data=preprocessed
uv run python -m sidon.train model=sidon_vocoder_finetune  data=preprocessed_48k \
    model.cfg.ssl_model_name=/path/to/feature_predictor.ckpt \
    model.cfg.pretrain_path=/path/to/vocoder_pretrain.ckpt

# Cleanse existing WebDataset shards with TorchScript checkpoints
uv run python scripts/cleanse_webdataset.py --help
```

There is no test suite, lint config, or formatter wired into this repo — don't fabricate one.

## Architecture

### Configuration (Hydra)
Two top-level entrypoints, two top-level configs:

- `config/config.yaml` → `sidon.train` (composes `data/`, `model/`, `train/` groups).
- `config/preprocess.yaml` → `sidon.preprocess` (composes `data/webdataset_preprocess_*` and `preprocess/default`).

Models, data modules, callbacks, and loggers are all instantiated via `hydra.utils.instantiate(...)` from `_target_` keys. To wire in new components, add a config under the appropriate group rather than editing the entrypoint scripts. Hydra writes per-run artefacts to `outputs/<timestamped_run>/` (logs, checkpoints, TensorBoard).

### Data flow
1. **Raw audio → WebDataset shards** via `sidon.preprocess` (uses `WebDatasetDataModule` from `src/sidon/data/preprocess/`). Augmentations (RIR, noise, codec) live in `degrations.py` / `functional_degrations.py`. A `Manager`/`Queue` fan-out drives N writer processes that each emit `worker-*-dataset-*.tar` files.
2. **Preprocessed shards → training** via `PreprocessedDataModule` (`src/sidon/data/datamodule.py`). Shards must contain `input_wav.pth`, `noisy_input_wav.pth`, optional `ssl_inputs.pickle` / `noisy_ssl_inputs.pickle`, and `sr.index`. Set `is_s3=true` in the data config to stream from S3 via the AWS CLI.
3. **Inference cleansing** via `scripts/cleanse_webdataset.py` and the `scripts/examples/cleanse_*.py` variants — these load TorchScripted feature extractor + decoder checkpoints (not Lightning checkpoints) and emit cleaned WebDataset shards.

### Models
Both Lightning modules live in `src/sidon/model/sidon/lightning_module.py`:

- `FeaturePredictorLightningModule` — student/teacher Wav2Vec2-BERT pair. Teacher is frozen on clean features; student gets LoRA adapters injected via `peft.inject_adapter_in_model` and is trained with MSE against the teacher's last hidden state from clean inputs.
- `SidonLightningModule` — the vocoder; wraps a DAC-derived decoder and discriminator with `DACLoss` + `GANLoss` from `src/sidon/model/losses.py`. Pretraining uses clean SSL features; finetuning swaps in the predicted (denoised) features and warm-starts from `model.cfg.pretrain_path`.

The three model configs in `config/model/` differ mainly in which Lightning module they target and which checkpoints they consume — they share the same `cfg` schema convention (`ssl_model_name`, `optim`, optionally `pretrain_path`).

### Cluster scripts
`scripts/pbs/*.sh` are PBS templates for the internal cluster. They expect `uv` on PATH, set `HF_HOME`, optionally use `S3_ENDPOINT_URL` for non-AWS S3, and rely on `PBS_ARRAY_INDEX`/`NUM_WORKERS` to shard work across array jobs. Cleansing scripts write to a local TMPDIR then `aws s3 sync` to the destination, using a `completed.txt` sentinel to make jobs idempotent.

## Things to know before editing

- Default config paths point at an internal cluster (`/groups/gag51394/...`). Always override `data.datamodule.train_urls` / `val_urls` / `preprocess.output_root` for local work — don't hardcode new paths.
- `s3fs` is pinned to `>=2023.9.0,<2023.11` and `datasets==2.17.1`. Be careful bumping these; the S3 streaming path is sensitive to the s3fs/fsspec version pair.
- The README explicitly notes the stack is "ported from an internal codebase and only partially smoke-checked." Treat unfamiliar behaviour as a likely real bug rather than assumed-correct.

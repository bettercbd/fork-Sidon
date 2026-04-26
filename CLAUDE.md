# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Sidon is a two-stage multilingual speech restoration system designed for large-scale dataset cleansing — converting noisy in-the-wild speech into studio-quality 48 kHz audio across 100+ languages. The architecture follows the parametric-resynthesis recipe used by Miipher / Miipher-2:

1. **Feature predictor** — LoRA-adapted w2v-BERT 2.0 that maps a noisy waveform's SSL representation toward the SSL representation of the corresponding clean speech.
2. **Vocoder** — HiFi-GAN-style decoder (DAC's snake-activation decoder) + discriminator that resynthesises a 48 kHz waveform from those denoised features.

The training and dataset recipe come from the paper at `2509.17052v3.pdf` (Nakata et al., "Sidon: Fast and Robust Open-Source Multilingual Speech Restoration for Large-Scale Dataset Cleansing"). A follow-up paper at `2604.09344v2.pdf` ("DialogueSidon") extends this stack with an SSL-VAE + diffusion latent predictor for joint dialogue restoration/separation; **that code is not in this repo** — only the single-speaker Sidon stack is implemented here.

The repo ships training, preprocessing, and dataset-cleansing pipelines built on PyTorch Lightning + Hydra, with WebDataset as the on-disk format.

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

- `FeaturePredictorLightningModule` — student/teacher Wav2Vec2-BERT pair. Both are loaded with `num_hidden_layers=8` (the paper takes the **8th-layer** hidden state of `facebook/w2v-bert-2.0` because earlier layers retain acoustic / speaker / prosody information needed for restoration, while later layers drift toward semantics). Teacher is frozen on clean features; student gets LoRA adapters (`r=64, α=16, dropout=0.1, target_modules=["output_dense"]`) injected via `peft.inject_adapter_in_model` and trained with MSE against the teacher's last hidden state. Of the ~198M total parameters, only ~5M are trainable (LoRA + bias).
- `SidonLightningModule` — the vocoder (~52.4M params). Wraps a DAC `Decoder` (`channels=1536`, `rates=[8, 5, 4, 3, 2]` → 960× upsample so the 50 Hz w2v-BERT features land at 48 kHz; `input_channel` is the SSL hidden size, 1024) and DAC discriminator with `DACLoss` (mel reconstruction) + `GANLoss` (adversarial + feature matching) from `src/sidon/model/losses.py`. `cfg.pretraining=True` consumes ground-truth clean SSL features through the frozen 8-layer w2v-BERT directly; `cfg.pretraining=False` instead loads the trained `FeaturePredictorLightningModule` from `cfg.ssl_model_name` (a Lightning `.ckpt`, **not** an HF model id) and warm-starts decoder/discriminator from `cfg.pretrain_path`. Optimisation is manual (`automatic_optimization=False`) because generator and discriminator alternate.

The three model configs in `config/model/` differ mainly in which Lightning module they target and which checkpoints they consume — they share the same `cfg` schema convention (`ssl_model_name`, `optim`, optionally `pretrain_path`).

### Training schedule (from the paper)
The published checkpoints were produced with this schedule on 8× H200 — useful as a sanity check when tuning configs:

| Stage | Steps | Wall time | Batch size | Notes |
| --- | --- | --- | --- | --- |
| Feature predictor | 400k | ~4 days | 256 | Uses the full 2,219 h corpus (24 kHz + 48 kHz) |
| Vocoder pretrain  | 140k | ~2 days | 32  | **48 kHz only** — mixing 24 kHz hurts fidelity |
| Vocoder finetune  | 280k | ~4 days | 32  | 48 kHz only; warm-started from pretrain |

AdamW (β1=0.9, β2=0.999), lr=1e-4, weight_decay=0.01. The vocoder uses an exponential LR decay with γ=0.9998. Note the defaults in `config/train/default.yaml` cap `max_steps` at 100k — bump it if you are reproducing the paper.

### Degradation pipeline (preprocess time)
Six independent degradations from `src/sidon/data/preprocess/degrations.py`, each applied with **p=0.5**, in this order: reverberation (pyroomacoustics RIR; RT60 ∈ U(0.1, 2.0) s, room dims ∈ U(2, 20) m), background noise (AudioSet / FMA / WHAM! / FSD50K / synthetic wind, SNR ∈ U(−5, 20) dB), band limitation (resample to one of {8, 16, 22.05, 24, 44.1, 48} kHz then back), clipping (quantile-based at percentiles 0–10 / 90–100), MP3 codec (65–245 kbps), packet loss (~9% of frames, segments 20–200 ms zeroed). The paper applies the pipeline 4× per clean utterance (≈9,000 h paired data from the 2,219 h source). If you change degradation distributions, expect generalisation behaviour to shift — the breadth of these six is one of the reasons Sidon matches Miipher-2 with a smaller SSL backbone.

### Cluster scripts
`scripts/pbs/*.sh` are PBS templates for the internal cluster. They expect `uv` on PATH, set `HF_HOME`, optionally use `S3_ENDPOINT_URL` for non-AWS S3, and rely on `PBS_ARRAY_INDEX`/`NUM_WORKERS` to shard work across array jobs. Cleansing scripts write to a local TMPDIR then `aws s3 sync` to the destination, using a `completed.txt` sentinel to make jobs idempotent.

## Things to know before editing

- Default config paths point at an internal cluster (`/groups/gag51394/...`). Always override `data.datamodule.train_urls` / `val_urls` / `preprocess.output_root` for local work — don't hardcode new paths.
- `s3fs` is pinned to `>=2023.9.0,<2023.11` and `datasets==2.17.1`. Be careful bumping these; the S3 streaming path is sensitive to the s3fs/fsspec version pair.
- The 8-layer truncation of w2v-BERT 2.0 and the LoRA `target_modules=["output_dense"]` setting are load-bearing design choices from the paper, not defaults to tweak casually — earlier-layer SSL features are what preserve speaker identity during restoration, and the LoRA scope is what keeps the multilingual pretraining intact.
- Vocoder finetuning expects `model.cfg.ssl_model_name` to be a **Lightning `.ckpt`** of a trained `FeaturePredictorLightningModule` (the constructor calls `FeaturePredictorLightningModule.load_from_checkpoint`). Pretraining uses an HF model id at the same key. Don't mix them up.
- Cleansing scripts (`scripts/cleanse_webdataset.py`, `scripts/examples/cleanse_*.py`) consume **TorchScripted** feature extractor + decoder checkpoints exported separately, not the raw Lightning checkpoints used for training.
- The README explicitly notes the stack is "ported from an internal codebase and only partially smoke-checked." Treat unfamiliar behaviour as a likely real bug rather than assumed-correct.

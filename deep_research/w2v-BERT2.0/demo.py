# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "torchaudio>=2.1",
#   "transformers>=4.44",
#   "numpy",
#   "soundfile>=0.12",
# ]
# ///
"""
w2v-BERT 2.0 capability demo.

Run:
    uv run --no-project deep_research/w2v-BERT2.0/demo.py
    uv run --no-project deep_research/w2v-BERT2.0/demo.py --audio /path/to/some.wav

What this prints:
  1. Model / parameter summary.
  2. Frame rate sanity check (16 kHz waveform -> 50 Hz SSL frames).
  3. Per-layer hidden-state L2 norms — shows representations evolving with depth.
  4. Per-layer temporal self-similarity (mean cosine between adjacent frames) —
     early/middle layers stay locally smooth (acoustic / phonetic), final layers
     become more "semantic" and frame-to-frame similarity often drops.
  5. Cosine similarity between layer 8 (the layer Sidon truncates to) and the
     final layer for the same utterance — illustrates *why* the paper picks an
     earlier layer: it preserves acoustic/speaker detail the vocoder needs.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel

MODEL_ID = "facebook/w2v-bert-2.0"
TARGET_SR = 16_000
SAMPLE_URL = (
    "https://download.pytorch.org/torchaudio/tutorial-assets/"
    "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)


def _fetch_sample() -> Path:
    cache_dir = Path.home() / ".cache" / "sidon-deep-research"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(SAMPLE_URL).name
    if not dest.exists():
        print(f"[setup] downloading sample audio to {dest}")
        urllib.request.urlretrieve(SAMPLE_URL, dest)
    return dest


def load_audio(path: str | None) -> tuple[torch.Tensor, int, str]:
    """Return (waveform[T], sample_rate, source_label). Mono, float32."""
    if path is None:
        asset = _fetch_sample()
        label = f"sample ({asset.name})"
        src = str(asset)
    else:
        label = path
        src = path

    data, sr = sf.read(src, dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR
    return wav.squeeze(0).contiguous(), sr, label


def fmt_n(n: int) -> str:
    return f"{n:,}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo basic capabilities of w2v-BERT 2.0.")
    ap.add_argument("--audio", default=None, help="Optional path to a WAV/FLAC file.")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: auto)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[setup] device = {device}")
    print(f"[setup] loading {MODEL_ID} (this downloads ~1.2 GB on first run)")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, output_hidden_states=True).to(device).eval()
    )

    total = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(
        f"[model] params={fmt_n(total)}  encoder_layers={n_layers}  "
        f"hidden_size={hidden}  feature_extractor_sr={extractor.sampling_rate}"
    )

    wav, sr, label = load_audio(args.audio)
    duration = wav.shape[-1] / sr
    print(f"[audio] {label}")
    print(f"[audio] samples={fmt_n(wav.shape[-1])}  sr={sr}  duration={duration:.2f}s")

    inputs = extractor(wav.numpy(), sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    last = out.last_hidden_state  # [1, T, H]
    hs = out.hidden_states  # tuple of [1, T, H], length = n_layers + 1
    T = last.shape[1]
    print(
        f"[forward] last_hidden_state shape = {tuple(last.shape)}  "
        f"=> {T} frames over {duration:.2f}s "
        f"≈ {T / max(duration, 1e-6):.1f} Hz (expected ~50 Hz)"
    )
    print(
        f"[forward] hidden_states tuple length = {len(hs)} (input embedding + {n_layers} layers)"
    )

    # 1) Per-layer L2 norm of the representation (mean over frames).
    print(
        "\n[layers] mean L2 norm per layer (input emb = 0, then encoder layers 1..N):"
    )
    norms = [h.squeeze(0).norm(dim=-1).mean().item() for h in hs]
    for i, n in enumerate(norms):
        bar = "█" * int(40 * n / max(norms))
        tag = " <-- Sidon layer 8" if i == 8 else ""
        print(f"  layer {i:>2}: {n:7.2f}  {bar}{tag}")

    # 2) Adjacent-frame cosine similarity per layer — acoustic locality vs. semantic drift.
    print(
        "\n[layers] mean cosine similarity between adjacent frames (1.0 = identical):"
    )
    for i, h in enumerate(hs):
        x = h.squeeze(0)  # [T, H]
        a = torch.nn.functional.normalize(x[:-1], dim=-1)
        b = torch.nn.functional.normalize(x[1:], dim=-1)
        sim = (a * b).sum(dim=-1).mean().item()
        bar = "█" * int(40 * max(sim, 0.0))
        tag = " <-- Sidon layer 8" if i == 8 else ""
        print(f"  layer {i:>2}: {sim:+.3f}  {bar}{tag}")

    # 3) How different is layer 8 from the final layer for the same utterance?
    if n_layers >= 8:
        l8 = torch.nn.functional.normalize(hs[8].squeeze(0), dim=-1)
        lf = torch.nn.functional.normalize(hs[-1].squeeze(0), dim=-1)
        per_frame = (l8 * lf).sum(dim=-1)
        print(
            f"\n[compare] cos(layer8, last_layer) per-frame: "
            f"mean={per_frame.mean().item():+.3f}  "
            f"min={per_frame.min().item():+.3f}  "
            f"max={per_frame.max().item():+.3f}"
        )
        print(
            "  -> low similarity means the two layers encode different things; "
            "Sidon keeps layer 8 because it retains speaker / prosody detail "
            "that the final layer has already abstracted away."
        )

    # 4) Tiny "what frames look alike?" peek: top-3 most similar frames to frame T//2 at layer 8.
    if n_layers >= 8 and T >= 8:
        x = torch.nn.functional.normalize(hs[8].squeeze(0), dim=-1)
        anchor_idx = T // 2
        sims = x @ x[anchor_idx]
        sims[anchor_idx] = -1.0  # exclude self
        topv, topi = torch.topk(sims, k=min(3, T - 1))
        anchor_t = anchor_idx / 50.0
        print(
            f"\n[neighbors] at layer 8, frame {anchor_idx} (~{anchor_t:.2f}s) "
            f"is most similar to:"
        )
        for v, i in zip(topv.tolist(), topi.tolist()):
            print(f"  frame {i:>4} (~{i / 50.0:.2f}s)  cos={v:+.3f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
❯ uv run --no-project deep_research/w2v-BERT2.0/demo.py
[setup] device = cpu
[setup] loading facebook/w2v-bert-2.0 (this downloads ~1.2 GB on first run)
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 773/773 [00:00<00:00, 8402.26it/s]
[model] params=580,493,120  encoder_layers=24  hidden_size=1024  feature_extractor_sr=16000
[audio] sample (Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav)
[audio] samples=54,400  sr=16000  duration=3.40s
[forward] last_hidden_state shape = (1, 169, 1024)  => 169 frames over 3.40s ≈ 49.7 Hz (expected ~50 Hz)
[forward] hidden_states tuple length = 25 (input embedding + 24 layers)

[layers] mean L2 norm per layer (input emb = 0, then encoder layers 1..N):
  layer  0:   65.35  ████████████████████████████████████████
  layer  1:   24.12  ██████████████
  layer  2:   24.92  ███████████████
  layer  3:   24.59  ███████████████
  layer  4:   21.30  █████████████
  layer  5:   26.62  ████████████████
  layer  6:   24.29  ██████████████
  layer  7:   22.45  █████████████
  layer  8:    7.76  ████ <-- Sidon layer 8
  layer  9:   25.52  ███████████████
  layer 10:   26.74  ████████████████
  layer 11:   26.98  ████████████████
  layer 12:   24.72  ███████████████
  layer 13:   26.49  ████████████████
  layer 14:   24.28  ██████████████
  layer 15:   25.50  ███████████████
  layer 16:   24.57  ███████████████
  layer 17:   22.15  █████████████
  layer 18:   21.94  █████████████
  layer 19:   20.30  ████████████
  layer 20:   19.58  ███████████
  layer 21:   17.24  ██████████
  layer 22:   14.70  ████████
  layer 23:   12.06  ███████
  layer 24:    5.09  ███

[layers] mean cosine similarity between adjacent frames (1.0 = identical):
  layer  0: +0.434  █████████████████
  layer  1: +0.711  ████████████████████████████
  layer  2: +0.738  █████████████████████████████
  layer  3: +0.753  ██████████████████████████████
  layer  4: +0.786  ███████████████████████████████
  layer  5: +0.746  █████████████████████████████
  layer  6: +0.737  █████████████████████████████
  layer  7: +0.751  ██████████████████████████████
  layer  8: +0.797  ███████████████████████████████ <-- Sidon layer 8
  layer  9: +0.682  ███████████████████████████
  layer 10: +0.603  ████████████████████████
  layer 11: +0.559  ██████████████████████
  layer 12: +0.573  ██████████████████████
  layer 13: +0.599  ███████████████████████
  layer 14: +0.638  █████████████████████████
  layer 15: +0.662  ██████████████████████████
  layer 16: +0.670  ██████████████████████████
  layer 17: +0.670  ██████████████████████████
  layer 18: +0.649  █████████████████████████
  layer 19: +0.645  █████████████████████████
  layer 20: +0.635  █████████████████████████
  layer 21: +0.659  ██████████████████████████
  layer 22: +0.683  ███████████████████████████
  layer 23: +0.695  ███████████████████████████
  layer 24: +0.838  █████████████████████████████████

[compare] cos(layer8, last_layer) per-frame: mean=+0.057  min=-0.076  max=+0.154
  -> low similarity means the two layers encode different things; Sidon keeps layer 8 because it retains speaker / prosody detail that the final layer has already abstracted away.

[neighbors] at layer 8, frame 84 (~1.68s) is most similar to:
  frame  125 (~2.50s)  cos=+0.756
  frame   85 (~1.70s)  cos=+0.750
  frame   91 (~1.82s)  cos=+0.731

Done.

"""

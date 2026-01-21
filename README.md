# caption-folder-mlx

A **local, offline image captioning CLI** for macOS Apple Silicon using **MLX-VLM**.
Designed for **dataset-quality captions**, including **accurate pose, clothing, actions**, and **neutral handling of adult / NSFW content**.

This tool was built and tested on **Apple M‑series (M2/M3/M4)** machines and is optimized for **stability, correctness, and reproducibility** rather than speed.

---

## Features

- 📂 Caption all images in a folder (optionally recursive)
- 🧠 Runs **fully locally** using MLX (no cloud, no API keys)
- 🎯 Strict, factual captions (no guessing; “unclear” allowed)
- 🧍 Uses `[trigger]` instead of gendered terms (LoRA / dataset‑safe)
- 🔞 Neutral, clinical handling of nudity and explicit content
- 📝 Writes captions as `.txt` files next to images
- 🏷️ Output filenames can include **full Hugging Face model ID + timestamp**
- 🔁 Optional **second-pass verification** for higher accuracy
- 💥 Optional **process isolation** to survive Metal OOM crashes

---

## Example Output

```
# model: mlx-community/Qwen2.5-VL-7B-Instruct-8bit
# timestamp: 20260121-140902
# image: 0001-08_1800.jpg

Pose: standing upright with weight on one leg, arms relaxed at sides, gaze forward
Clothes: black crop top and matching shorts
Action: [trigger] is standing still facing the camera
NSFW: lingerie
Scene: indoor studio with plain background
```

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install mlx mlx-vlm pillow
```

> ⚠️ **Do NOT install PyTorch** — this tool is MLX-only.

---

## Supported Models (MLX‑Compatible)

✅ Recommended:

- `mlx-community/Qwen2.5-VL-7B-Instruct-8bit` (default)
- `mlx-community/Qwen2-VL-7B-Instruct`
- `mlx-community/InternVL2-8B-MLX` (best accuracy, heavier)

🚫 Not supported:

- “NSFW Caption” models
- `mxfp4` / CUDA-only checkpoints
- Any model not explicitly converted for MLX

---

## Basic Usage

```bash
python caption_folder_mlx.py ./images
```

This will:
- Caption every image in `./images`
- Write `<image>__<model>__<timestamp>.txt` next to each image

---

## Common Options

### Specify model

```bash
--model mlx-community/Qwen2.5-VL-7B-Instruct-8bit
```

### Resize images (recommended)

```bash
--max-side 1280
```

Lower this if you hit memory issues:
```bash
--max-side 1024
```

### Control generation

```bash
--max-tokens 170
--temperature 0.15
```

---

## Overwrite Behavior

By default, existing `.txt` files are skipped.

```bash
--overwrite
```

---

## Fixed Output Name (no model/timestamp)

```bash
--fixed-name
```

Output:
```
image.jpg → image.txt
```

Metadata is still written **inside** the file.

---

## Accuracy Mode (Recommended)

Enable a **second verification pass** where the model checks and corrects its own caption:

```bash
--verify
```

This significantly reduces:
- Hallucinated clothing
- Incorrect poses
- Wrong NSFW classification

---

## Stability Mode (Metal OOM Safe)

If you see errors like:

```
[METAL] Command buffer execution failed: Insufficient Memory
```

Run with **process isolation**:

```bash
--isolate
```

Each image is processed in its own subprocess, so GPU crashes do not kill the batch.

---

## Recursive Folders

```bash
--recursive
```

---

## `[trigger]` Enforcement

All captions:
- Use `[trigger]` instead of person / man / woman / etc.
- As a final safety net, the script replaces common human nouns automatically

To disable this (not recommended):

```bash
--no-replace-nouns
```

---

## What This Tool Is For

- Stable Diffusion / Flux / LoRA dataset captioning
- NSFW dataset labeling (neutral, non‑erotic)
- Pose / clothing / action extraction
- Offline, reproducible ML workflows on Apple Silicon

---

## What This Tool Is NOT

- A chatbot
- A content generator
- A moderation classifier
- Fast batch inference at scale

This prioritizes **correctness over speed**.

---

## Troubleshooting

### PyTorch error

```
ImportError: PyTorch is not installed
```

✅ This is expected.  
❌ If it crashes, you are using a non‑MLX model.

---

### Model loads but fails with “Missing parameters”

The model is **not MLX‑converted**. Use only `mlx-community/*` models.

---

### Captions are inaccurate

- Enable `--verify`
- Lower `--temperature`
- Switch to `InternVL2-8B-MLX`

---

## License

This script is provided as-is for research and dataset preparation purposes.
Model licenses are governed by their respective Hugging Face repositories.

---

## Author Notes

Built specifically for:
- Apple Silicon (M‑series)
- Local AI workflows
- Dataset quality > speed

If you want:
- JSON/CSV output
- Confidence scores
- Multi-subject support
- LoRA token injection

…the script is designed to be easily extended.

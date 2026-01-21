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

## Distributed Mode (Remote MLX Machines)

For large datasets, distribute captioning across **multiple remote MLX machines** via SSH:

```bash
python caption_folder_mlx_remote.py ./images --config hosts.json
```

This will:
- Connect to each configured MLX machine via SSH (password-authenticated)
- Distribute images evenly across machines
- Copy images to remote, generate captions, copy results back
- Maintain the same skip logic and caption format
- Clean up remote temporary files automatically

### Setup

### Setup

1. **Create a `hosts.json` config** (see `hosts_example.json`):
   ```json
   {
     "hosts": [
       {
         "hostname": "mlx-machine-1.example.com",
         "username": "user",
         "password": "your-password",
         "port": 22,
         "remote_work_dir": "/tmp/caption_work",
         "script_path": "/opt/caption-folder-mlx/caption_folder_mlx.py"
       },
       {
         "hostname": "mlx-machine-2.example.com",
         "username": "user",
         "password": "your-password",
         "port": 22,
         "remote_work_dir": "/tmp/caption_work_2",
         "script_path": "/opt/caption-folder-mlx/caption_folder_mlx.py"
       }
     ]
   }
   ```

2. **Ensure remote machines have Python 3.11.14 installed**:
   ```bash
   # Check if available
   python3.11 --version
   
   # If not installed, the setup will fail with clear instructions
   ```

3. **Just run the script** - it handles everything else:
   ```bash
   python caption_folder_mlx_remote.py ./images --config hosts.json
   ```

The script will automatically:
- ✅ Install local dependencies (`paramiko`, `tqdm`) if needed
- ✅ Create virtual environments on all remote machines
- ✅ Install all MLX dependencies on remote machines
- ✅ Deploy the caption script to remote machines
- ✅ Verify everything before starting captioning

### Options

All local script options work with the remote version:

```bash
--model mlx-community/Qwen2.5-VL-7B-Instruct-8bit
--max-tokens 170
--temperature 0.15
--max-side 1280
--verify
--fixed-name
--overwrite
--recursive
--no-replace-nouns
```

Example:
```bash
python caption_folder_mlx_remote.py ./large_dataset \
  --config hosts.json \
  --model mlx-community/InternVL2-8B-MLX \
  --verify \
  --max-side 1024
```

### Features

- ✅ **Skip already-captioned images** (same model)
- ✅ **Progress bar** with current status per remote machine
- ✅ **Automatic cleanup** of temporary files on remote
- ✅ **SFTP file transfer** (efficient binary copying)
- ✅ **Load balancing** distributes images evenly across hosts
- ✅ **Error resilience** continues if one machine fails

### Performance

With 4 MLX machines processing in parallel, you can expect ~4x throughput compared to single-machine mode.

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

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

**On macOS (Apple Silicon) or Linux:**
```bash
pip install --upgrade pip
pip install mlx mlx-vlm pillow tqdm
```

**On Windows (for remote mode only):**
```powershell
python caption_folder_mlx_remote.py ./images --config hosts.json
```

The script will automatically install `paramiko` and `tqdm` locally if needed!

---

## System Requirements

### Local Host (where you run the script)
- **macOS (Apple Silicon M1/M2/M3/M4+)** or **Linux** (required for MLX)
  - **Python 3.10+** 
  - `mlx`, `mlx-vlm`, `Pillow`, `tqdm`
- **Windows**: Use distributed mode only (see below)
  - **Python 3.10+**
  - `paramiko` for remote SSH connections
  - `tqdm` for progress bars
  - ⚠️ Cannot run local captioning on Windows (MLX not supported)

### Remote Machines (MLX captioning)
- **Linux** (recommended) or **macOS Apple Silicon**
- **Python 3.11.14** (specifically for MLX compatibility)
- `mlx`, `mlx-vlm`, `Pillow`, `tqdm`

> ⚠️ **Important**: Remote machines MUST have Python 3.11.14. The automated setup will verify this.

---

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

### Local Mode (macOS Apple Silicon / Linux only)

```bash
python caption_folder_mlx.py ./images
```

This will:
- Caption every image in `./images`
- Write `<image>__<model>__<timestamp>.txt` next to each image

> ⚠️ **Not available on Windows** — use distributed mode instead.

### Distributed Mode (Windows, macOS, Linux)

Connect to multiple remote MLX machines to distribute the work:

```bash
python caption_folder_mlx_remote.py ./images --config hosts.json
```

This works from **any OS** (Windows, macOS, Linux) to orchestrate remote MLX machines.

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

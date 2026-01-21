#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"

DEFAULT_PROMPT = """Describe the image with emphasis on:
- Pose/body position (standing/sitting, limb positions, orientation, gaze)
- Clothing & accessories (type, colors, layers, fit, footwear, jewelry)
- Actions/interactions (what the person is doing/holding/touching)
Be concrete. Avoid guessing identity. If something is unclear, say "unclear".
Return 2-4 sentences.
"""

def iter_images(folder: Path, recursive: bool):
    if recursive:
        yield from (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        yield from (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing images")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id/path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    print(f"Loading MLX-VLM model: {args.model}")
    model, processor = load(args.model)
    config = load_config(args.model)

    imgs = list(iter_images(folder, args.recursive))
    if not imgs:
        print("No images found.")
        return

    for img_path in imgs:
        out_path = img_path.with_suffix(".txt")
        if out_path.exists() and not args.overwrite:
            print(f"SKIP (exists): {out_path.name}")
            continue

        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            formatted = apply_chat_template(processor, config, args.prompt, num_images=1)

            # generate() accepts PIL images in a list (per MLX-VLM docs/examples)
            caption = generate(
                model,
                processor,
                formatted,
                [img],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=False,
            ).strip()

            out_path.write_text(caption + "\n", encoding="utf-8")
            print(f"OK: {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"ERR: {img_path.name}: {e}")

if __name__ == "__main__":
    main()

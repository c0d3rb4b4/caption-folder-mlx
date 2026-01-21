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

DEFAULT_PROMPT = """You are captioning an image for dataset creation.

IMPORTANT RULES:
- Refer to the human subject ONLY as "[trigger]".
- DO NOT use words like person, woman, man, girl, boy, model, subject, individual.
- If multiple people appear, still describe the primary subject as "[trigger]".
- Never infer identity, age, or name.

Describe the image with emphasis on:
Pose: body position, limb placement, posture, orientation, gaze.
Clothes: garments, colors, materials, layers, fit, footwear, accessories.
Action: what [trigger] is doing, holding, touching, or interacting with.
Scene: brief environment/context.

Return exactly 4 lines in this format:
Pose: ...
Clothes: ...
Action: ...
Scene: ...
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
            result = generate(
                model,
                processor,
                formatted,
                [img],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=False,
            )

            # mlx-vlm may return either a string or a GenerationResult-like object
            if isinstance(result, str):
                caption = result
            else:
                # common attribute names across versions
                caption = (
                    getattr(result, "text", None)
                    or getattr(result, "output_text", None)
                    or getattr(result, "generated_text", None)
                )
                if caption is None:
                    # last resort: try dict-like or repr
                    try:
                        caption = result["text"]
                    except Exception:
                        caption = str(result)

            caption = caption.strip()

            # Hard replace common human nouns just in case
            REPLACEMENTS = [
                " person", " woman", " man", " girl", " boy",
                " Person", " Woman", " Man", " Girl", " Boy",
                " model", " Model", " subject", " Subject"
            ]

            for w in REPLACEMENTS:
                caption = caption.replace(w, " [trigger]")

            out_path.write_text(caption + "\n", encoding="utf-8")
            print(f"OK: {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"ERR: {img_path.name}: {e}")

if __name__ == "__main__":
    main()

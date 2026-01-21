#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from platform import processor
from unittest import result
from xml.parsers.expat import model

from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import re
from datetime import datetime

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"

DEFAULT_PROMPT = """You are generating a factual caption for an image dataset.

CRITICAL RULES:
- Describe ONLY what is clearly visible. Do not guess missing details.
- If a detail is uncertain, write "unclear" rather than inventing.
- Refer to the main human subject ONLY as "[trigger]". Do not use person/woman/man/girl/boy/model/subject.
- If the image contains nudity or sexual content, describe it clinically and neutrally (e.g., "nudity", "lingerie", "explicit sexual activity"), without erotic language.

Return exactly 5 lines:
Pose: (posture, limb placement, orientation, gaze)
Clothes: (garments, colors, layers; if nude, say "nudity" and what is visible)
Action: (what [trigger] is doing/holding/touching/interacting with)
NSFW: (none | nudity | lingerie | explicit)
Scene: (brief environment/context)
"""

def iter_images(folder: Path, recursive: bool):
    if recursive:
        yield from (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        yield from (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def downscale(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)

def hf_model_id_to_filename(model_id: str) -> str:
    """
    Convert a Hugging Face model id into a filesystem-safe,
    but still HF-searchable, filename component.
    """
    # Replace "/" with "_" so org/model is preserved
    s = model_id.replace("/", "_")

    # Remove anything truly unsafe, keep dots and dashes
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    return s

def extract_text(result):
    if isinstance(result, str):
        return result
    for attr in ("text", "output_text", "generated_text"):
        v = getattr(result, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    try:
        v = result["text"]
        if isinstance(v, str):
            return v
    except Exception:
        pass
    return str(result)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing images")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id/path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-side", type=int, default=1280,
                help="Resize so the longest image side is <= this (0 disables).")

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
        model_tag = hf_model_id_to_filename(args.model)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        out_path = img_path.with_name(
            f"{img_path.stem}__{model_tag}__{ts}.txt"
        )

        if out_path.exists() and not args.overwrite:
            print(f"SKIP (exists): {out_path.name}")
            continue
        
        header = (
                f"# model: {args.model}\n"
                f"# timestamp: {ts}\n"
                f"# image: {img_path.name}\n\n"
            )

        out_path.write_text(header + caption + "\n", encoding="utf-8")
        
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            img = downscale(img, args.max_side)

            formatted = apply_chat_template(processor, config, args.prompt, num_images=1)

            # PASS 1: initial caption
            result1 = generate(model, processor, formatted, [img],
                   max_tokens=args.max_tokens, temperature=args.temperature, verbose=False)
            
            caption1 = extract_text(result1).strip()

            # PASS 2: verification/correction
            verify_prompt = f"""You are validating a caption against the image.

            Caption to verify:
            {caption1}

            Instructions:
            - Check each line for accuracy against the image.
            - Remove any incorrect or uncertain details.
            - Replace unknowns with "unclear".
            - Keep the same 5-line format (Pose/Clothes/Action/NSFW/Scene).
            - Main subject must be called [trigger].

            Now output the corrected caption only.
            """

            formatted2 = apply_chat_template(processor, config, verify_prompt, num_images=1)
            result2 = generate(model, processor, formatted2, [img],
                            max_tokens=args.max_tokens, temperature=0.0, verbose=False)

            caption = extract_text(result2).strip()

            # mlx-vlm may return either a string or a GenerationResult-like object
            if isinstance(result1, str):
                caption = result1
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
            #REPLACEMENTS = [
            #    " person", " woman", " man", " girl", " boy",
            #    " Person", " Woman", " Man", " Girl", " Boy",
            #    " model", " Model", " subject", " Subject"
            #]

            #for w in REPLACEMENTS:
            #    caption = caption.replace(w, " [trigger]")

            out_path.write_text(caption + "\n", encoding="utf-8")

            print(f"OK: {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"ERR: {img_path.name}: {e}")

if __name__ == "__main__":
    main()

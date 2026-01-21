#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"

DEFAULT_PROMPT = """You are generating a factual caption for an image dataset.

CRITICAL RULES:
- Do NOT replace visible genital anatomy with euphemisms such as "inner thighs" or "private area".
- Precision is preferred over vagueness when visibility is clear.
- Describe ONLY what is clearly visible. Do not guess missing details.
- If a detail is uncertain, write "unclear" rather than inventing.
- Refer to the main human subject ONLY as "[trigger]". Do not use person/woman/man/girl/boy/model/subject.
- If the image contains nudity or sexual content, describe it clinically and anatomically accurately.
- Use correct anatomical terms (e.g., vulva, labia, penis, breasts) ONLY when they are clearly visible.
- Do not euphemize anatomy that is clearly shown.
- If anatomy is partially visible or ambiguous, say "unclear".

Return exactly 5 lines:
Pose: (posture, limb placement, orientation, gaze)
Clothes: (garments, colors, layers; if nude, say "nudity" and what is visible)
Action: (what [trigger] is doing with their body or hands; be anatomically accurate if visible)
NSFW: (none | nudity | lingerie | explicit)
Scene: (brief environment/context)
"""


def iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
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
    Filesystem-safe, still HuggingFace-searchable:
      org/model -> org_model
    Keep letters, digits, dot, underscore, dash.
    """
    s = model_id.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def extract_text(gen_result) -> str:
    """
    mlx_vlm.generate() returns a GenerationResult in recent versions.
    We only accept actual text-like fields. Never fall back to str(gen_result),
    because that can silently write garbage.
    """
    if isinstance(gen_result, str) and gen_result.strip():
        return gen_result

    for attr in ("text", "output_text", "generated_text"):
        v = getattr(gen_result, attr, None)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(v).strip()
            if joined:
                return joined

    # dict-like
    try:
        v = gen_result.get("text")  # type: ignore[attr-defined]
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(v).strip()
            if joined:
                return joined
    except Exception:
        pass

    raise TypeError(
        f"Could not extract caption text from generate() result of type {type(gen_result)}. "
        f"Available attrs include: {[a for a in dir(gen_result) if 'text' in a.lower()]}"
    )


_TRIGGER_WORD_RE = re.compile(
    r"\b(person|woman|man|girl|boy|model|subject)\b",
    flags=re.IGNORECASE,
)

def enforce_trigger_words(text: str) -> str:
    """
    Last-resort cleanup: replace common human nouns with [trigger].
    Keep it conservative: whole words only.
    """
    return _TRIGGER_WORD_RE.sub("[trigger]", text)


def build_header(model_id: str, ts: str, image_name: str) -> str:
    return f"# model: {model_id}\n# timestamp: {ts}\n# image: {image_name}\n\n"


def make_output_path(img_path: Path, model_id: str, ts: str, fixed_name: bool) -> Path:
    """
    - fixed_name=True  -> <stem>.txt (next to image)
    - fixed_name=False -> <stem>__<org_model>__<timestamp>.txt
    """
    if fixed_name:
        return img_path.with_suffix(".txt")
    model_tag = hf_model_id_to_filename(model_id)
    return img_path.with_name(f"{img_path.stem}__{model_tag}__{ts}.txt")


def caption_one_inprocess(
    model,
    processor,
    config,
    img_path: Path,
    prompt: str,
    max_side: int,
    max_tokens: int,
    temperature: float,
    verify: bool,
    replace_nouns: bool,
) -> str:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = downscale(img, max_side)

    # IMPORTANT: apply_chat_template ONCE (in-process), with exactly one image.
    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    r1 = generate(
        model,
        processor,
        formatted,
        [img],
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )
    caption1 = extract_text(r1).strip()

    if verify:
        verify_prompt = f"""You are validating a caption against the image.

Caption to verify:
{caption1}

Instructions:
- Replace any euphemisms with anatomically accurate terms if anatomy is clearly visible.
- Do not introduce new details.
- If the original caption avoided specificity incorrectly, correct it.

Now output the corrected caption only.
"""
        formatted2 = apply_chat_template(processor, config, verify_prompt, num_images=1)
        r2 = generate(
            model,
            processor,
            formatted2,
            [img],
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )
        caption = extract_text(r2).strip()
    else:
        caption = caption1

    if replace_nouns:
        caption = enforce_trigger_words(caption)

    return caption.strip()


def run_isolated_child(knowing_venv_python: str, argv: list[str]) -> Tuple[int, str, str]:
    """
    Run a single-image job in a subprocess so a Metal OOM won't kill the whole batch.
    """
    p = subprocess.run(
        [knowing_venv_python, *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Caption images in a folder using MLX-VLM; writes .txt next to images."
    )
    ap.add_argument("folder", nargs="?", help="Folder containing images")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id/path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")

    # Output naming
    ap.add_argument(
        "--fixed-name",
        action="store_true",
        help="Write <image_stem>.txt (no model/timestamp in filename). Metadata still goes inside file.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists.",
    )

    # Generation controls
    ap.add_argument("--max-tokens", type=int, default=170)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1280, help="Resize so longest side <= this (0 disables).")
    ap.add_argument("--verify", action="store_true", help="Second pass: verify & correct against image (slower, more accurate).")
    ap.add_argument(
        "--no-replace-nouns",
        action="store_true",
        help="Do not apply last-resort replacement of person/man/woman/etc -> [trigger].",
    )

    # Stability option
    ap.add_argument(
        "--isolate",
        action="store_true",
        help="Process each image in its own subprocess (survives Metal OOMs; slower).",
    )

    # Internal: single-image worker mode (used by --isolate)
    ap.add_argument("--_single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_image", help=argparse.SUPPRESS)

    args = ap.parse_args()

    replace_nouns = not args.no_replace_nouns

    # --- Isolated worker mode: do exactly one image and print caption to stdout ---
    if args._single:
        if not args._image:
            raise SystemExit("Internal error: --_single requires --_image")
        img_path = Path(args._image).expanduser().resolve()
        if not img_path.exists():
            raise SystemExit(f"Image not found: {img_path}")

        model, processor = load(args.model)
        config = load_config(args.model)

        cap = caption_one_inprocess(
            model=model,
            processor=processor,
            config=config,
            img_path=img_path,
            prompt=args.prompt,
            max_side=args.max_side,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verify=args.verify,
            replace_nouns=replace_nouns,
        )
        # Print only the caption text; parent will add header + write file.
        print(cap)
        return

    # --- Normal batch mode ---
    if not args.folder:
        ap.print_help()
        raise SystemExit(2)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    images = sorted(iter_images(folder, args.recursive))
    if not images:
        print("No images found.")
        return

    # Load model once unless we isolate per image
    model = processor = config = None
    if not args.isolate:
        print(f"Loading MLX-VLM model: {args.model}")
        model, processor = load(args.model)
        config = load_config(args.model)

    venv_python = sys.executable

    for img_path in images:
        ts = now_tag()
        out_path = make_output_path(img_path, args.model, ts, fixed_name=args.fixed_name)

        if out_path.exists() and not args.overwrite:
            print(f"SKIP (exists): {out_path.name}")
            continue

        header = build_header(args.model, ts, img_path.name)

        try:
            if args.isolate:
                # Spawn child that loads model and captions one image
                child_argv = [
                    str(Path(__file__).resolve()),
                    "--_single",
                    "--_image", str(img_path),
                    "--model", args.model,
                    "--prompt", args.prompt,
                    "--max-side", str(args.max_side),
                    "--max-tokens", str(args.max_tokens),
                    "--temperature", str(args.temperature),
                ]
                if args.verify:
                    child_argv.append("--verify")
                if args.fixed_name:
                    child_argv.append("--fixed-name")
                if args.overwrite:
                    child_argv.append("--overwrite")
                if args.no_replace_nouns:
                    child_argv.append("--no-replace-nouns")

                rc, stdout, stderr = run_isolated_child(venv_python, child_argv)
                if rc != 0:
                    raise RuntimeError(f"child failed (rc={rc}). stderr:\n{stderr.strip()}\nstdout:\n{stdout.strip()}")
                caption = stdout.strip()
                if not caption:
                    raise RuntimeError(f"child returned empty caption. stderr:\n{stderr.strip()}")
            else:
                assert model is not None and processor is not None and config is not None
                caption = caption_one_inprocess(
                    model=model,
                    processor=processor,
                    config=config,
                    img_path=img_path,
                    prompt=args.prompt,
                    max_side=args.max_side,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verify=args.verify,
                    replace_nouns=replace_nouns,
                )

            out_path.write_text(header + caption + "\n", encoding="utf-8")
            print(f"OK: {img_path.name} -> {out_path.name}")

        except Exception as e:
            print(f"ERR: {img_path.name}: {e}")


if __name__ == "__main__":
    main()

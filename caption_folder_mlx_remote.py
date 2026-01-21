#!/usr/bin/env python3
"""
caption_folder_mlx_remote.py

Distributed image captioning across remote MLX machines via SSH.

- Copies one image at a time to each host's /tmp work dir
- Runs caption_folder_mlx.py remotely inside a per-host venv
- Copies the generated .txt back immediately (safe resume)

Pinned deps are used to avoid common mlx-vlm/transformers incompatibilities.
"""
from __future__ import annotations

import argparse
import json
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional

import paramiko
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
REMOTE_DEP_VERSIONS = [
    ("mlx", "0.30.3"),
    ("mlx-vlm", "0.3.9"),
    ("mlx-lm", "0.23.2"),
    ("transformers", "4.51.3"),
    ("Pillow", "12.1.0"),
    ("tqdm", "4.67.1"),
]
REMOTE_DEP_VERSION_MAP = {name: version for name, version in REMOTE_DEP_VERSIONS}
REMOTE_DEP_SPECS = [
    f"{name}=={version}"
    for name, version in REMOTE_DEP_VERSIONS
    if name != "mlx-vlm"
]
REMOTE_VLM_SPEC = "mlx-vlm==0.3.9"


@dataclass
class RemoteHost:
    hostname: str
    username: str
    password: str
    port: int = 22
    remote_work_dir: str = "/tmp/caption_work_1"
    script_path: str = ""


def iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        yield from (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def model_to_safe_id(model_id: str) -> str:
    s = model_id.replace("/", "_")
    s = "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def caption_file_exists(img_path: Path, model_id: str, fixed_name: bool) -> tuple[bool, Optional[Path]]:
    if fixed_name:
        expected = img_path.with_suffix(".txt")
        return (expected.exists(), expected if expected.exists() else None)
    safe = model_to_safe_id(model_id)
    matches = sorted(img_path.parent.glob(f"{img_path.stem}__{safe}__*.txt"))
    return (len(matches) > 0, matches[0] if matches else None)


def build_version_check_cmd(python_bin: str) -> str:
    req_repr = repr(REMOTE_DEP_VERSION_MAP)
    code = "\n".join(
        [
            "import importlib.metadata as md",
            f"req = {req_repr}",
            "bad = []",
            "for name, expected in req.items():",
            "    try:",
            "        v = md.version(name)",
            "    except md.PackageNotFoundError:",
            "        bad.append(f\"{name} missing\")",
            "        continue",
            "    if v != expected:",
            "        bad.append(f\"{name} {v} != {expected}\")",
            "if bad:",
            "    print(\"\\n\".join(bad))",
            "    raise SystemExit(1)",
            "print(\"deps_ok\")",
        ]
    )
    return f"{python_bin} -c {shlex.quote(code)}"


def build_version_summary_cmd(python_bin: str) -> str:
    names = [name for name, _ in REMOTE_DEP_VERSIONS]
    code = "\n".join(
        [
            "import importlib.metadata as md",
            f"names = {repr(names)}",
            "out = []",
            "for name in names:",
            "    try:",
            "        out.append(f\"{name} {md.version(name)}\")",
            "    except md.PackageNotFoundError:",
            "        out.append(f\"{name} missing\")",
            "print('; '.join(out))",
        ]
    )
    return f"{python_bin} -c {shlex.quote(code)}"


def ssh_connect(host: RemoteHost) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host.hostname, port=host.port, username=host.username, password=host.password, timeout=30)
    return c


def remote_exec(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def remote_mkdir(client: paramiko.SSHClient, path: str) -> None:
    rc, _, err = remote_exec(client, f"mkdir -p '{path}'")
    if rc != 0:
        raise RuntimeError(f"Failed to create remote dir {path}: {err.strip()}")


def copy_to_remote(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def copy_from_remote(client: paramiko.SSHClient, remote_path: str, local_path: Path) -> None:
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def load_hosts_config(config_file: Path) -> list[RemoteHost]:
    data = json.loads(config_file.read_text(encoding="utf-8"))
    return [RemoteHost(**h) for h in data.get("hosts", [])]


def find_python_311(client: paramiko.SSHClient) -> str:
    candidates = [
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.11",
        "/usr/bin/python3.11",
        "python3.11",
        "python3",
    ]
    for c in candidates:
        rc, out, _ = remote_exec(client, f"{c} --version 2>&1")
        if rc == 0 and "3.11" in out:
            return c
    raise RuntimeError("Python 3.11 not found on remote host.")


def setup_remote_host(client: paramiko.SSHClient, host: RemoteHost, local_caption_script: Path) -> None:
    py = find_python_311(client)
    rc, out, err = remote_exec(client, f"{py} --version 2>&1")
    if rc != 0:
        raise RuntimeError(err.strip())
    tqdm.write(f"[{host.hostname}] Found {out.strip()} at: {py}")

    remote_mkdir(client, host.remote_work_dir)
    venv_dir = f"{host.remote_work_dir}/.venv"
    pyvenv = f"{venv_dir}/bin/python"
    pip = f"{venv_dir}/bin/pip"

    rc, _, _ = remote_exec(client, f"test -f '{pyvenv}'")
    if rc != 0:
        tqdm.write(f"[{host.hostname}] Creating venv...")
        rc, _, err = remote_exec(client, f"{py} -m venv '{venv_dir}'")
        if rc != 0:
            raise RuntimeError(err.strip())

    rc, out, err = remote_exec(client, build_version_check_cmd(pyvenv))
    if rc != 0:
        details = (out or err).strip()
        if details:
            tqdm.write(f"[{host.hostname}] Deps mismatch:\n{details}")
        tqdm.write(f"[{host.hostname}] Installing deps (pinned)...")
        rc, _, err = remote_exec(client, f"{pip} install -U pip setuptools wheel")
        if rc != 0:
            raise RuntimeError(err.strip())
        joined = " ".join(shlex.quote(r) for r in REMOTE_DEP_SPECS)
        rc, _, err = remote_exec(client, f"{pip} install -U {joined}")
        if rc != 0:
            raise RuntimeError(err.strip())
        # Install mlx-vlm without deps to keep transformers pinned to a known-good version.
        rc, _, err = remote_exec(client, f"{pip} install -U --no-deps {shlex.quote(REMOTE_VLM_SPEC)}")
        if rc != 0:
            raise RuntimeError(err.strip())
        rc, out, err = remote_exec(client, build_version_check_cmd(pyvenv))
        if rc != 0:
            raise RuntimeError((out or err).strip())
    else:
        tqdm.write(f"[{host.hostname}] Deps already match pinned versions")

    remote_script = f"{host.remote_work_dir}/caption_folder_mlx.py"
    copy_to_remote(client, local_caption_script, remote_script)
    host.script_path = remote_script

    rc, out, err = remote_exec(client, build_version_summary_cmd(pyvenv))
    if rc != 0:
        raise RuntimeError(err.strip())
    tqdm.write(f"[{host.hostname}] Setup OK: {out.strip()}")


def process_worker(host: RemoteHost, q: Queue, pbar: tqdm, args) -> None:
    client = None
    try:
        client = ssh_connect(host)
        venv_dir = f"{host.remote_work_dir}/.venv"
        py = f"{venv_dir}/bin/python"

        while True:
            img_path = q.get()
            if img_path is None:
                break

            remote_img = f"{host.remote_work_dir}/{img_path.name}"
            try:
                copy_to_remote(client, img_path, remote_img)

                cmd = (
                    f"{py} {shlex.quote(host.script_path)} {shlex.quote(host.remote_work_dir)}"
                    f" --model {shlex.quote(args.model)}"
                    f" --max-side {args.max_side}"
                    f" --max-tokens {args.max_tokens}"
                    f" --temperature {args.temperature}"
                )
                if args.prompt:
                    cmd += f" --prompt {shlex.quote(args.prompt)}"
                if args.overwrite:
                    cmd += " --overwrite"
                if args.fixed_name:
                    cmd += " --fixed-name"
                if args.verify:
                    cmd += " --verify"
                if args.no_replace_nouns:
                    cmd += " --no-replace-nouns"

                rc, out, err = remote_exec(client, cmd)
                if rc != 0:
                    tqdm.write(f"ERR: {host.hostname} - {img_path.name}: {err.strip() or out.strip()}")
                    remote_exec(client, f"rm -f '{remote_img}'")
                    pbar.update(1)
                    continue

                if args.fixed_name:
                    remote_cap = f"{host.remote_work_dir}/{img_path.stem}.txt"
                else:
                    safe = model_to_safe_id(args.model)
                    rc2, out2, _ = remote_exec(
                        client,
                        f"ls -t '{host.remote_work_dir}/{img_path.stem}__{safe}__'*.txt 2>/dev/null | head -1",
                    )
                    remote_cap = out2.strip() if rc2 == 0 else ""

                if not remote_cap:
                    tqdm.write(f"ERR: {host.hostname} - {img_path.name}: no caption output found")
                    remote_exec(client, f"rm -f '{remote_img}'")
                    pbar.update(1)
                    continue

                local_cap = img_path.with_suffix(".txt") if args.fixed_name else (img_path.parent / Path(remote_cap).name)
                copy_from_remote(client, remote_cap, local_cap)

                remote_exec(client, f"rm -f '{remote_img}' '{remote_cap}'")

                pbar.set_postfix_str(f"{host.hostname}: ✓ {local_cap.name}")
                pbar.update(1)

            except Exception as e:
                tqdm.write(f"ERR: {host.hostname} - {img_path.name}: {e}")
                pbar.update(1)

    finally:
        if client:
            client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Distributed MLX image captioning over SSH.")
    ap.add_argument("folder", help="Local folder containing images")
    ap.add_argument("--config", required=True, help="JSON config file with hosts")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", help="Optional prompt override passed to caption_folder_mlx.py")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--fixed-name", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=170)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-replace-nouns", action="store_true")

    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    config = Path(args.config).expanduser().resolve()
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    hosts = load_hosts_config(config)
    if not hosts:
        raise SystemExit("No hosts in config")

    local_caption_script = Path(__file__).parent / "caption_folder_mlx.py"
    if not local_caption_script.exists():
        raise SystemExit(f"caption_folder_mlx.py not found next to remote script: {local_caption_script}")

    imgs = sorted(iter_images(folder, args.recursive))
    if not imgs:
        print("No images found.")
        return

    to_process = []
    for p in imgs:
        if not args.overwrite:
            exists, _ = caption_file_exists(p, args.model, args.fixed_name)
            if exists:
                continue
        to_process.append(p)

    print(f"Found {len(imgs)} image(s) to scan ({len(imgs)-len(to_process)} already captioned).")
    if not to_process:
        print("Nothing to do.")
        return

    print("\nSetting up remote hosts...")
    for h in hosts:
        c = ssh_connect(h)
        setup_remote_host(c, h, local_caption_script)
        c.close()
        tqdm.write(f"[{h.hostname}] Ready")

    q: Queue = Queue()
    for p in to_process:
        q.put(p)
    for _ in hosts:
        q.put(None)

    pbar = tqdm(total=len(to_process), desc="Remote captioning", unit="img")
    threads = []
    for h in hosts:
        t = threading.Thread(target=process_worker, args=(h, q, pbar, args), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    pbar.close()

    print("\nDistributed captioning complete!")


if __name__ == "__main__":
    main()

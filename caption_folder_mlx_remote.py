#!/usr/bin/env python3
"""
Distributed image captioning across remote MLX machines via SSH.

Orchestrates distribution of images to multiple MLX-capable machines,
captures generated captions, and saves them back on the host.
Maintains skip logic across all remote sessions.
Processes images in parallel across all hosts with immediate caption write-back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from dataclasses import dataclass, asdict
from queue import Queue

import paramiko
from tqdm import tqdm


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


@dataclass
class RemoteHost:
    """Configuration for a remote MLX machine."""
    hostname: str
    username: str
    password: str
    port: int = 22
    remote_work_dir: str = "/tmp/caption_work"
    script_path: str = "/opt/caption-folder-mlx/caption_folder_mlx.py"


def iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
    """Iterate over image files in folder."""
    if recursive:
        yield from (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        yield from (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def caption_file_exists(img_path: Path, model_id: str, fixed_name: bool) -> tuple[bool, Optional[Path]]:
    """
    Check if caption already exists for this image and model.
    Returns (exists, path) where path is the existing file or None.
    """
    if fixed_name:
        expected = img_path.with_suffix(".txt")
        if expected.exists():
            return True, expected
        return False, None
    
    # For timestamped files, we need to check for any existing caption with this model
    from re import escape
    model_safe = model_id.replace("/", "_")
    model_safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_safe).strip("_")
    pattern = f"{img_path.stem}__{model_safe}__*.txt"
    parent = img_path.parent
    
    matches = list(parent.glob(pattern))
    if matches:
        return True, matches[0]
    return False, None


def ssh_connect(host: RemoteHost) -> paramiko.SSHClient:
    """Create SSH connection to remote host."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host.hostname,
        port=host.port,
        username=host.username,
        password=host.password,
        timeout=30,
    )
    return client


def remote_exec(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    """Execute command on remote host and return (returncode, stdout, stderr)."""
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8")
    err = stderr.read().decode("utf-8")
    returncode = stdout.channel.recv_exit_status()
    return returncode, out, err


def remote_mkdir(client: paramiko.SSHClient, path: str) -> None:
    """Create directory on remote host."""
    rc, _, err = remote_exec(client, f"mkdir -p {path}")
    if rc != 0:
        raise RuntimeError(f"Failed to create remote directory {path}: {err}")


def copy_to_remote(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    """Copy file from local to remote via SFTP."""
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def copy_from_remote(client: paramiko.SSHClient, remote_path: str, local_path: Path) -> None:
    """Copy file from remote to local via SFTP."""
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def load_hosts_config(config_file: Path) -> list[RemoteHost]:
    """Load remote hosts from JSON config file."""
    with open(config_file) as f:
        data = json.load(f)
    
    hosts = []
    for item in data.get("hosts", []):
        hosts.append(RemoteHost(**item))
    return hosts


def save_hosts_config(config_file: Path, hosts: list[RemoteHost]) -> None:
    """Save remote hosts to JSON config file."""
    data = {
        "hosts": [asdict(h) for h in hosts]
    }
    with open(config_file, "w") as f:
        json.dump(data, f, indent=2)


def find_python_311(client: paramiko.SSHClient) -> str:
    """
    Try multiple ways to find Python 3.11.
    Returns the command to use for Python 3.11 or raises error.
    """
    # Try different potential paths
    candidates = [
        "python3.11",
        "/usr/bin/python3.11",
        "/usr/local/bin/python3.11",
        "/opt/homebrew/bin/python3.11",  # macOS Apple Silicon
        "python3",  # Fallback - check if it's 3.11
    ]
    
    for candidate in candidates:
        rc, stdout, _ = remote_exec(client, f"{candidate} --version 2>&1")
        if rc == 0 and "3.11" in stdout:
            return candidate
    
    # Last resort: try with bash login shell
    rc, stdout, _ = remote_exec(client, "bash -lc 'python3.11 --version 2>&1'")
    if rc == 0 and "3.11" in stdout:
        return "bash -lc 'python3.11"  # Special handling needed
    
    raise RuntimeError("Python 3.11 not found. Tried: " + ", ".join(candidates))


def setup_remote_host(
    client: paramiko.SSHClient,
    host: RemoteHost,
    local_script_path: Path,
) -> None:
    """
    Set up remote host with all requirements:
    1. Check Python 3.11.14
    2. Create virtual environment
    3. Install dependencies
    4. Copy caption script to remote
    """
    # Find Python 3.11
    python_bin = find_python_311(client)
    
    # If it was the bash login shell version, extract just the base command
    if "bash -lc" in python_bin:
        python_base = "python3.11"
        rc, stdout, stderr = remote_exec(client, f"bash -lc '{python_base} --version'")
    else:
        python_base = python_bin
        rc, stdout, stderr = remote_exec(client, f"{python_bin} --version")
    
    python_version = stdout.strip()
    tqdm.write(f"[{host.hostname}] Found {python_version} at: {python_base}")
    
    # Verify it's 3.11.x
    if "3.11" not in python_version:
        raise RuntimeError(f"Python 3.11 required on {host.hostname}, but found: {python_version}")
    
    # Create directories
    remote_mkdir(client, host.remote_work_dir)
    venv_dir = f"{host.remote_work_dir}/.venv"
    
    # Check if venv already exists and has all requirements
    check_venv = f"test -f {venv_dir}/bin/python && {venv_dir}/bin/python -c 'import mlx_vlm, PIL, tqdm' 2>/dev/null"
    rc, _, _ = remote_exec(client, check_venv)
    
    if rc == 0:
        tqdm.write(f"[{host.hostname}] Virtual environment already set up with all requirements")
    else:
        tqdm.write(f"[{host.hostname}] Setting up virtual environment with Python 3.11...")
        
        # Create virtual environment using found Python 3.11
        rc, _, err = remote_exec(client, f"{python_base} -m venv {venv_dir}")
        if rc != 0:
            raise RuntimeError(f"Failed to create venv on {host.hostname}: {err}")
        
        tqdm.write(f"[{host.hostname}] Installing dependencies...")
        
        # Upgrade pip and install dependencies
        pip_cmd = f"{venv_dir}/bin/pip install --upgrade pip setuptools wheel"
        rc, _, err = remote_exec(client, pip_cmd)
        if rc != 0:
            raise RuntimeError(f"Failed to upgrade pip on {host.hostname}: {err}")
        
        # Install required packages with specific versions
        # Use latest mlx-vlm with compatible transformers for Pixtral/Paligemma
        deps = [
            "mlx>=0.0.14",
            "mlx-vlm>=0.1.0",  # Latest version
            "transformers>=4.45.0",  # Required for newer models
            "Pillow>=10.0.0",
            "tqdm>=4.66.0",
        ]
        
        for dep in deps:
            install_cmd = f"{venv_dir}/bin/pip install '{dep}'"
            rc, _, err = remote_exec(client, install_cmd)
            if rc != 0:
                raise RuntimeError(f"Failed to install {dep} on {host.hostname}: {err}")
            tqdm.write(f"[{host.hostname}] Installed {dep}")
    
    # Copy caption script to remote if needed
    # Deploy script to the work directory (more reliable than parent directory)
    expected_script_path = f"{host.remote_work_dir}/caption_folder_mlx.py"
    
    # Check if script exists
    rc, _, _ = remote_exec(client, f"test -f {expected_script_path}")
    if rc != 0:
        if not local_script_path.exists():
            raise RuntimeError(f"Local script not found: {local_script_path}")
        
        tqdm.write(f"[{host.hostname}] Copying caption script...")
        # Ensure work directory exists
        remote_mkdir(client, host.remote_work_dir)
        try:
            copy_to_remote(client, local_script_path, expected_script_path)
        except Exception as e:
            raise RuntimeError(f"Failed to copy script to {host.hostname}: {e}")
        
        tqdm.write(f"[{host.hostname}] Script deployed to {expected_script_path}")
    else:
        tqdm.write(f"[{host.hostname}] Script already deployed")
    
    # Always use the deployed script path in work directory
    host.script_path = expected_script_path
    
    # Verify setup
    verify_cmd = f"{venv_dir}/bin/python -c 'import mlx_vlm, PIL; print(\"Setup verified\")'"
    rc, stdout, err = remote_exec(client, verify_cmd)
    if rc != 0:
        raise RuntimeError(f"Setup verification failed on {host.hostname}: {err}")
    
    tqdm.write(f"[{host.hostname}] Setup complete and verified")


def process_worker(
    host: RemoteHost,
    image_queue: Queue,
    pbar: tqdm,
    args,
    fixed_name: bool,
) -> None:
    """Worker thread that processes images from queue on a single remote host."""
    try:
        client = ssh_connect(host)
        venv_dir = f"{host.remote_work_dir}/.venv"
        python_bin = f"{venv_dir}/bin/python"
        remote_mkdir(client, host.remote_work_dir)
        
        while True:
            # Get next image from shared queue (None signals end)
            img_path = image_queue.get()
            if img_path is None:
                break
            
            try:
                pbar.set_postfix_str(f"Processing on {host.hostname}: {img_path.name}")
                
                # Copy image to remote
                remote_img_path = f"{host.remote_work_dir}/{img_path.name}"
                copy_to_remote(client, img_path, remote_img_path)
                
                # Build remote caption command using venv python
                remote_cmd = f"{python_bin} {host.script_path} {host.remote_work_dir}"
                remote_cmd += f" --model '{args.model}'"
                remote_cmd += f" --max-tokens {args.max_tokens}"
                remote_cmd += f" --temperature {args.temperature}"
                remote_cmd += f" --max-side {args.max_side}"
                
                if args.fixed_name:
                    remote_cmd += " --fixed-name"
                if args.verify:
                    remote_cmd += " --verify"
                if args.no_replace_nouns:
                    remote_cmd += " --no-replace-nouns"
                
                # Run captioning on remote
                rc, stdout, stderr = remote_exec(client, remote_cmd)
                
                if rc != 0:
                    tqdm.write(f"ERR: {host.hostname} - {img_path.name}: {stderr.strip()}")
                    pbar.update(1)
                    continue
                
                # Find the generated caption file on remote
                list_cmd = f"ls -t {host.remote_work_dir}/{img_path.stem}*.txt 2>/dev/null | head -1"
                rc, remote_cap_path, _ = remote_exec(client, list_cmd)
                
                if rc != 0 or not remote_cap_path.strip():
                    tqdm.write(f"ERR: {host.hostname} - {img_path.name}: No caption file generated")
                    pbar.update(1)
                    continue
                
                remote_cap_path = remote_cap_path.strip()
                
                # Determine local output path
                if args.fixed_name:
                    local_cap_path = img_path.with_suffix(".txt")
                else:
                    # Extract just the filename from remote path
                    remote_filename = remote_cap_path.split("/")[-1]
                    local_cap_path = img_path.parent / remote_filename
                
                # Copy caption back to host (IMMEDIATE write-back)
                copy_from_remote(client, remote_cap_path, local_cap_path)
                
                # Cleanup remote files
                remote_exec(client, f"rm -f {remote_img_path} {remote_cap_path}")
                
                pbar.set_postfix_str(f"✓ Generated: {local_cap_path.name}")
                pbar.update(1)
                
            except Exception as e:
                tqdm.write(f"ERR: {host.hostname} - {img_path.name}: {e}")
                pbar.update(1)
        
        client.close()
        
    except Exception as e:
        tqdm.write(f"ERR: Failed to connect to {host.hostname}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Distribute image captioning across remote MLX machines via SSH."
    )
    ap.add_argument("folder", nargs="?", help="Local folder containing images")
    ap.add_argument("--config", required=True, help="JSON config file with remote host credentials")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id/path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--fixed-name", action="store_true", help="Write <image_stem>.txt")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--max-tokens", type=int, default=170)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1280, help="Resize so longest side <= this")
    ap.add_argument("--verify", action="store_true", help="Second pass verification")
    ap.add_argument("--no-replace-nouns", action="store_true", help="Skip noun replacement")

    args = ap.parse_args()

    # Check and install local dependencies
    print("Checking local dependencies...")
    try:
        import paramiko
        from tqdm import tqdm as _  # Just check if tqdm is available
    except ImportError:
        print("Installing required local packages (paramiko, tqdm)...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "tqdm", "-q"])
        print("Local dependencies installed!")
    
    if not args.folder:
        ap.print_help()
        raise SystemExit(2)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    config_file = Path(args.config).expanduser().resolve()
    if not config_file.exists():
        raise SystemExit(f"Config file not found: {config_file}")

    # Load remote hosts
    try:
        hosts = load_hosts_config(config_file)
    except Exception as e:
        raise SystemExit(f"Failed to load config: {e}")

    if not hosts:
        raise SystemExit("No hosts configured in config file")

    # Verify local script exists
    script_dir = Path(__file__).parent
    local_script = script_dir / "caption_folder_mlx.py"
    if not local_script.exists():
        raise SystemExit(f"Local script not found: {local_script}")

    # Collect images
    images = sorted(iter_images(folder, args.recursive))
    if not images:
        print("No images found.")
        return

    # Filter out images that already have captions (unless --overwrite)
    images_to_process = []
    for img_path in images:
        if not args.overwrite:
            exists, _ = caption_file_exists(img_path, args.model, args.fixed_name)
            if exists:
                continue
        images_to_process.append(img_path)

    total_images = len(images_to_process)
    print(f"Found {total_images} image(s) to process ({len(images) - total_images} already captioned).")

    if total_images == 0:
        print("All images already have captions.")
        return

    # Setup all remote hosts
    print("\nSetting up remote hosts...")
    for host in hosts:
        try:
            client = ssh_connect(host)
            setup_remote_host(client, host, local_script)
            client.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise SystemExit(f"Failed to setup {host.hostname}: {e}")

    print("All remote hosts ready!\n")

    # Create shared queue for images to process
    image_queue: Queue = Queue()
    for img_path in images_to_process:
        image_queue.put(img_path)
    
    # Add sentinel values (None) for each worker to signal end of work
    for _ in hosts:
        image_queue.put(None)
    
    # Progress tracking
    pbar = tqdm(total=total_images, desc="Remote captioning", unit="img")

    try:
        # Start worker thread for each host
        threads = []
        for host in hosts:
            thread = threading.Thread(
                target=process_worker,
                args=(host, image_queue, pbar, args, args.fixed_name),
                daemon=False,
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
    finally:
        pbar.close()

    print("\nDistributed captioning complete!")


if __name__ == "__main__":
    main()

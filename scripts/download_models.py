#!/usr/bin/env python
"""Auto-download the RAM-tier local AI model (Section 11 tiers).

    ≥14 GB  microsoft/Phi-3-mini-4k-instruct-gguf   Q4_K_M
    ≥ 7 GB  TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF Q4_K_M
    else    Qwen/Qwen2-0.5B-Instruct-GGUF          Q4_K_M

Always exits 0 on SKIP (offline / hub unavailable) so the automated
setup flow keeps going; prints a machine-readable status line.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:") or line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    try:
        import subprocess
        out = subprocess.run(["free", "-m"], capture_output=True, text=True,
                             timeout=3).stdout
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ram", type=float, default=None,
                    help="override detected RAM (GB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print which model would be fetched")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file exists")
    args = ap.parse_args()

    from app.agents import LOCAL_MODELS, MODELS_DIR

    ram = args.ram if args.ram is not None else ram_gb()
    repo, fname, label = None, None, None
    for min_ram, r, f, lab in LOCAL_MODELS:
        if ram >= min_ram:
            repo, fname, label = r, f, lab
            break
    assert repo and fname and label
    path = os.path.join(MODELS_DIR, fname)

    print(f"[model] detected RAM: {ram:.1f} GB -> tier: {label}")
    if args.dry_run:
        print(f"[model] DRY-RUN would fetch {repo}/{fname} -> {path}")
        return 0
    if os.path.exists(path) and not args.force:
        print(f"[model] DONE already present: {path}")
        return 0

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:
        print(f"[model] SKIPPED huggingface_hub unavailable ({e.__class__.__name__}); "
              "run: pip install -r requirements-agents.txt")
        return 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[model] downloading {repo}/{fname} …")
    try:
        got = hf_hub_download(repo_id=repo, filename=fname, local_dir=MODELS_DIR)
        print(f"[model] DONE -> {got}")
        return 0
    except Exception as e:
        reason = e.__class__.__name__
        print(f"[model] SKIPPED download failed ({reason}) — offline or hub "
              "unreachable; retried automatically on first model use")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Auto-download the preferred local AI model (app.agents tiers).

    >= 5 GB  Qwen/Qwen2.5-Coder-3B-Instruct-GGUF  Q4_K_M  (preferred)
    else     Qwen/Qwen2-0.5B-Instruct-GGUF       Q4_K_M

Always exits 0 on SKIP (offline / hub unavailable) so setup flows
keep going; prints a machine-readable status line.
"""
from __future__ import annotations

import argparse
import os
import sys

# fail fast when the Hub is unreachable (auto-start must never stall)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "8")
import logging as _logging
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

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
    from app.agents import model_validation, write_model_sidecar
    if os.path.exists(path) and not args.force:
        v = model_validation(path)
        if v["magic_ok"]:
            if v["sha"].startswith("unverified"):
                write_model_sidecar(path)   # record checksum once
            print(f"[model] DONE already present: {path} "
                  f"({v['bytes'] / 1e9:.2f} GB, sha {model_validation(path)['sha']}, "
                  f"0 bytes fetched)")
            return 0
        print(f"[model] existing file is corrupt (not a GGUF) — re-fetching")
        os.remove(path)

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
        rec = write_model_sidecar(got)
        print(f"[model] DONE -> {got} (sha256 {rec['sha256'][:12]}…, "
              f"{rec['size'] / 1e9:.2f} GB)")
        return 0
    except Exception as e:
        reason = e.__class__.__name__
        print(f"[model] SKIPPED download failed ({reason}) — offline or hub "
              "unreachable; retried automatically on first model use")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Health probe that depends on NOTHING but python3 (stdlib).

The boot chain used to shell out to curl for every health check.
In a fresh Codespace that made the whole starter blind whenever
curl was absent/slow: a healthy engine looked dead, got killed,
restarted, forever — while the lock appeared held for a day.
python3 is our own runtime; it is guaranteed to exist.

    http_check.py URL [timeout_s]   exit 0 iff HTTP 2xx
"""
import sys
import urllib.request
import urllib.error


def main() -> int:
    url = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 0 if 200 <= r.status < 300 else 1
    except urllib.error.HTTPError as e:
        return 0 if 200 <= e.code < 300 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

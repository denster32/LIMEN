#!/usr/bin/env python3
"""Encrypted export/import for LIMEN state."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def run_openssl(args: list[str]) -> None:
    result = subprocess.run(["openssl", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "openssl command failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("--file", required=True, help="Encrypted archive path")
    parser.add_argument("--passphrase", required=True, help="Encryption passphrase")
    args = parser.parse_args()

    target = Path(args.file)
    if args.command == "export":
        run_openssl([
            "enc",
            "-aes-256-cbc",
            "-pbkdf2",
            "-salt",
            "-in",
            str(STATE_PATH),
            "-out",
            str(target),
            "-pass",
            f"pass:{args.passphrase}",
        ])
        print(f"exported {target}")
        return 0

    run_openssl([
        "enc",
        "-d",
        "-aes-256-cbc",
        "-pbkdf2",
        "-in",
        str(target),
        "-out",
        str(STATE_PATH),
        "-pass",
        f"pass:{args.passphrase}",
    ])
    print(f"imported {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

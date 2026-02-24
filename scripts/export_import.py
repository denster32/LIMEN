#!/usr/bin/env python3
"""Encrypted export/import for LIMEN state backups."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def _derive_keys(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    material = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000, dklen=64)
    return material[:32], material[32:]


def _xor_stream(data: bytes, key: bytes) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[: len(data)]))


def export_state(output_path: Path, passphrase: str) -> None:
    plaintext = STATE_PATH.read_bytes()
    salt = os.urandom(16)
    enc_key, mac_key = _derive_keys(passphrase, salt)
    ciphertext = _xor_stream(plaintext, enc_key)
    tag = hmac.new(mac_key, salt + ciphertext, hashlib.sha256).digest()
    blob = {
        "kdf": "pbkdf2_hmac_sha256",
        "iterations": 200000,
        "salt": base64.b64encode(salt).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "hmac_sha256": base64.b64encode(tag).decode("ascii"),
    }
    output_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")


def import_state(input_path: Path, passphrase: str) -> None:
    blob = json.loads(input_path.read_text(encoding="utf-8"))
    salt = base64.b64decode(blob["salt"])
    ciphertext = base64.b64decode(blob["ciphertext"])
    tag = base64.b64decode(blob["hmac_sha256"])
    enc_key, mac_key = _derive_keys(passphrase, salt)
    expected = hmac.new(mac_key, salt + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Backup integrity check failed (wrong passphrase or tampered file)")
    plaintext = _xor_stream(ciphertext, enc_key)
    parsed = json.loads(plaintext.decode("utf-8"))
    STATE_PATH.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    export_p = sub.add_parser("export")
    export_p.add_argument("--output", required=True)
    export_p.add_argument("--passphrase", required=True)

    import_p = sub.add_parser("import")
    import_p.add_argument("--input", required=True)
    import_p.add_argument("--passphrase", required=True)

    args = parser.parse_args()

    if args.cmd == "export":
        export_state(Path(args.output), args.passphrase)
    else:
        import_state(Path(args.input), args.passphrase)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

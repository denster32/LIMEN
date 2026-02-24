#!/usr/bin/env python3
"""Encrypted export/import for LIMEN state portability."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import secrets
from pathlib import Path

from scripts.state_ops import load_state, save_state


def _derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    key_material = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000, dklen=64)
    return key_material[:32], key_material[32:]


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _encrypt(plaintext: bytes, password: str) -> dict[str, str]:
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    enc_key, mac_key = _derive_keys(password, salt)
    cipher = bytes(a ^ b for a, b in zip(plaintext, _keystream(enc_key, nonce, len(plaintext))))
    tag = hmac.new(mac_key, nonce + cipher, hashlib.sha256).digest()
    return {
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(cipher).decode(),
        "tag": base64.b64encode(tag).decode(),
    }


def _decrypt(blob: dict[str, str], password: str) -> bytes:
    salt = base64.b64decode(blob["salt"])
    nonce = base64.b64decode(blob["nonce"])
    cipher = base64.b64decode(blob["ciphertext"])
    tag = base64.b64decode(blob["tag"])
    enc_key, mac_key = _derive_keys(password, salt)
    expected = hmac.new(mac_key, nonce + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Integrity check failed; wrong password or tampered backup")
    return bytes(a ^ b for a, b in zip(cipher, _keystream(enc_key, nonce, len(cipher))))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--output", required=True)
    ex.add_argument("--password", required=True)

    im = sub.add_parser("import")
    im.add_argument("--input", required=True)
    im.add_argument("--password", required=True)

    args = parser.parse_args()
    if args.mode == "export":
        blob = _encrypt(json.dumps(load_state(), ensure_ascii=False).encode("utf-8"), args.password)
        Path(args.output).write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
        print(f"Exported encrypted backup to {args.output}")
    else:
        blob = json.loads(Path(args.input).read_text(encoding="utf-8"))
        state = json.loads(_decrypt(blob, args.password).decode("utf-8"))
        save_state(state)
        print("Imported encrypted backup into state/limen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

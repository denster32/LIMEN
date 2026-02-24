#!/usr/bin/env python3
"""Encrypted export for state/limen.json."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import secrets
from hashlib import pbkdf2_hmac, sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block = sha256(key + nonce + counter.to_bytes(4, "big")).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def encrypt(plaintext: bytes, passphrase: str) -> dict:
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    key = pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000, dklen=32)
    stream = _keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(key, nonce + ciphertext, "sha256").digest()
    return {
        "kdf": "pbkdf2-sha256",
        "iterations": 200000,
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "tag": base64.b64encode(tag).decode(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output encrypted export file")
    parser.add_argument("--passphrase", required=True)
    args = parser.parse_args()

    payload = STATE_PATH.read_bytes()
    encrypted = encrypt(payload, args.passphrase)
    Path(args.out).write_text(json.dumps(encrypted, indent=2) + "\n", encoding="utf-8")
    print(f"Encrypted export written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

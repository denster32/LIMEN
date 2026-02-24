#!/usr/bin/env python3
"""Import encrypted backup into state/limen.json."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
from hashlib import pbkdf2_hmac, sha256
from pathlib import Path

from scripts.validate_state import validate_state

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


def decrypt(payload: dict, passphrase: str) -> bytes:
    salt = base64.b64decode(payload["salt"])
    nonce = base64.b64decode(payload["nonce"])
    ciphertext = base64.b64decode(payload["ciphertext"])
    tag = base64.b64decode(payload["tag"])
    key = pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, payload.get("iterations", 200000), dklen=32)
    expected = hmac.new(key, nonce + ciphertext, "sha256").digest()
    if not hmac.compare_digest(expected, tag):
        raise ValueError("Backup authentication failed (wrong passphrase or corrupted file)")
    stream = _keystream(key, nonce, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--passphrase", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    plain = decrypt(payload, args.passphrase)
    state = json.loads(plain.decode("utf-8"))
    validate_state(state)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("state/limen.json restored from encrypted backup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

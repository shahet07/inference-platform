#!/usr/bin/env python3
"""Creates an API key. The key goes to stdout once, and the line for the key file goes to stderr.

Usage: make_api_key.py NAME [client|admin] [RATE] [BURST] >key.txt 2>>api_keys.txt
A rate of 0 means no limit.
"""
import hashlib
import secrets
import sys


def main():
    if not 2 <= len(sys.argv) <= 5:
        sys.exit(__doc__)
    name = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else "client"
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    burst = float(sys.argv[4]) if len(sys.argv) > 4 else max(rate, 1)
    if role not in ("client", "admin") or any(c.isspace() for c in name) or not name:
        sys.exit(__doc__)
    key = secrets.token_urlsafe(32)
    print(key)
    print(f"{hashlib.sha256(key.encode()).hexdigest()} {name} {role} {rate:g} {burst:g}", file=sys.stderr)


if __name__ == "__main__":
    main()

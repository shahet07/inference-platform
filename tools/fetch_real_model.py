#!/usr/bin/env python3
"""Downloads MobileNetV2 from the ONNX Model Zoo (Apache-2.0), quantized and float, and checks the hashes.
They are not committed, so they stay local in bench/models/.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/"
MODELS = {
    "mobilenetv2-12-int8.onnx": "cc028fe6cae7bc11a4ff53cfc9b79c920e8be65ce33a904ec3e2a8f66d77f95f",
    "mobilenetv2-12.onnx": "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
}
OUT = Path(__file__).resolve().parent.parent / "bench" / "models"

for name, expected in MODELS.items():
    path = OUT / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        print(f"Already have {path}")
        continue
    with urllib.request.urlopen(BASE + name, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        sys.exit(f"Refusing {name}: expected hash {expected}, got {digest}")
    path.write_bytes(data)
    print(f"Wrote {path}")

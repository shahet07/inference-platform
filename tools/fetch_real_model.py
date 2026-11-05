#!/usr/bin/env python3
"""Downloads a real quantized image model, MobileNetV2 from the ONNX Model Zoo (Apache-2.0), and
checks its hash. It is not committed, so bench/models/mobilenetv2-12-int8.onnx stays local.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/"
       "mobilenetv2-12-int8.onnx")
SHA256 = "cc028fe6cae7bc11a4ff53cfc9b79c920e8be65ce33a904ec3e2a8f66d77f95f"
OUT = Path(__file__).resolve().parent.parent / "bench" / "models" / "mobilenetv2-12-int8.onnx"

if OUT.exists() and hashlib.sha256(OUT.read_bytes()).hexdigest() == SHA256:
    print(f"Already have {OUT}")
    sys.exit(0)
with urllib.request.urlopen(URL, timeout=120) as response:
    data = response.read()
digest = hashlib.sha256(data).hexdigest()
if digest != SHA256:
    sys.exit(f"Refusing the download: expected hash {SHA256}, got {digest}")
OUT.write_bytes(data)
print(f"Wrote {OUT}")

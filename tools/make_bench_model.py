#!/usr/bin/env python3
"""Regenerates the benchmark models: MLPs in light and heavy sizes, float and quantized.

The weights are random but seeded. The model exists to produce a repeatable compute load, not
useful predictions.

Usage: pip install onnx onnxruntime numpy && python3 tools/make_bench_model.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

OUT = Path(__file__).resolve().parent.parent / "bench" / "models"
SIZES = {"mlp": [64, 256, 256, 10], "mlp_heavy": [512, 2048, 2048, 10],
         "mlp_xl": [1024, 4096, 4096, 10]}  # About 21 MB, so it is generated, not committed.


def build(name, layers):
    rng = np.random.default_rng(42)
    nodes, inits = [], []
    prev = "x"
    for i, (a, b) in enumerate(zip(layers, layers[1:])):
        inits.append(numpy_helper.from_array(rng.normal(0, 0.1, (a, b)).astype(np.float32), f"w{i}"))
        inits.append(numpy_helper.from_array(rng.normal(0, 0.1, (b,)).astype(np.float32), f"b{i}"))
        nodes.append(helper.make_node("MatMul", [prev, f"w{i}"], [f"m{i}"]))
        nodes.append(helper.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"]))
        prev = f"a{i}"
        if i < len(layers) - 2:
            nodes.append(helper.make_node("Relu", [prev], [f"r{i}"]))
            prev = f"r{i}"
    nodes.append(helper.make_node("Identity", [prev], ["y"]))
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", layers[0]])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", layers[-1]])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    float_path = OUT / f"{name}_float.onnx"
    onnx.save(model, float_path)
    quantize_dynamic(float_path, OUT / f"{name}_quant.onnx", weight_type=QuantType.QUInt8)
    if name != "mlp":
        float_path.unlink()  # Too large to keep, and only the quantized form is benchmarked.


OUT.mkdir(parents=True, exist_ok=True)
for model_name, model_layers in SIZES.items():
    build(model_name, model_layers)
print(f"Wrote models to {OUT}")

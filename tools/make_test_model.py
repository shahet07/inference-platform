#!/usr/bin/env python3
"""Regenerates tests/data/linear.onnx, a tiny model computing y = 2x + 1.

Usage: pip install onnx && python3 tools/make_test_model.py
"""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "linear.onnx"

graph = helper.make_graph(
    nodes=[
        helper.make_node("Mul", ["x", "slope"], ["scaled"]),
        helper.make_node("Add", ["scaled", "intercept"], ["y"]),
    ],
    name="linear",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 1])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 1])],
    initializer=[
        helper.make_tensor("slope", TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor("intercept", TensorProto.FLOAT, [], [1.0]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, OUTPUT)
print(f"Wrote {OUTPUT}")

#!/usr/bin/env python3
"""Writes copies of a model with a fixed batch size, since accelerators often need a static shape.

Usage: make_static_batch_model.py INPUT.onnx OUTPUT_PREFIX BATCH [BATCH...]
Needs: pip install onnx
"""
import sys

import onnx


def main():
    source, prefix = sys.argv[1], sys.argv[2]
    for batch in map(int, sys.argv[3:]):
        model = onnx.load(source)
        initializers = {i.name for i in model.graph.initializer}
        for value in [*model.graph.input, *model.graph.output]:
            if value.name in initializers:
                continue
            value.type.tensor_type.shape.dim[0].ClearField("dim_param")
            value.type.tensor_type.shape.dim[0].dim_value = batch
        onnx.save(model, f"{prefix}-{batch}.onnx")


if __name__ == "__main__":
    main()

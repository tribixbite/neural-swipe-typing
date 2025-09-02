#!/usr/bin/env python3
import onnx
import json

# Load the encoder model
encoder_model = onnx.load("../deployment_package/swipe_model_character.onnx")

print("Encoder Model Inputs:")
for input in encoder_model.graph.input:
    print(f"  Name: {input.name}")
    print(f"  Type: {input.type}")
    shape = []
    for dim in input.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
    print(f"  Shape: {shape}")
    print(f"  Element type: {input.type.tensor_type.elem_type}")
    print()

print("\nEncoder Model Outputs:")
for output in encoder_model.graph.output:
    print(f"  Name: {output.name}")
    print(f"  Type: {output.type}")
    shape = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
    print(f"  Shape: {shape}")
    print()

# Load the decoder model
decoder_model = onnx.load("../deployment_package/swipe_decoder_character.onnx")

print("\nDecoder Model Inputs:")
for input in decoder_model.graph.input:
    print(f"  Name: {input.name}")
    print(f"  Type: {input.type}")
    shape = []
    for dim in input.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
    print(f"  Shape: {shape}")
    print(f"  Element type: {input.type.tensor_type.elem_type}")
    print()

print("\nDecoder Model Outputs:")
for output in decoder_model.graph.output:
    print(f"  Name: {output.name}")
    print(f"  Type: {output.type}")
    shape = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
    print(f"  Shape: {shape}")
    print()

# Element type mapping
elem_types = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128"
}

print("\nElement type legend:")
for key, value in elem_types.items():
    print(f"  {key}: {value}")
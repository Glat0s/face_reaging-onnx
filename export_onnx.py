"""
Export the face re-aging UNet model to ONNX format.

Usage:
    python export_onnx.py --model_path model_files/best_unet_model.pth --output model_files/unet.onnx
"""

import argparse
import sys
import torch

sys.path.append(".")
from model.models import UNet


def export(model_path: str, output_path: str, opset: int = 17) -> None:
    device = torch.device("cpu")

    print(f"Loading weights from {model_path} ...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Input: (batch, 5, 512, 512)  — 3 RGB channels + source_age + target_age
    dummy_input = torch.randn(1, 5, 512, 512, device=device)

    print(f"Exporting to {output_path}  (opset {opset}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        dynamo=False,           # use legacy TorchScript exporter (avoids onnxscript bugs)
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Verify the exported graph
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("ONNX graph check passed.")
    except ImportError:
        print("onnx package not installed — skipping graph check.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export UNet to ONNX")
    parser.add_argument("--model_path", default="model_files/best_unet_model.pth")
    parser.add_argument("--output",     default="model_files/unet.onnx")
    parser.add_argument("--opset",      type=int, default=17)
    args = parser.parse_args()

    export(args.model_path, args.output, args.opset)

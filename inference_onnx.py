"""
Face re-aging inference using an ONNX Runtime session.

The ONNX model is a drop-in replacement for the PyTorch UNet; it accepts the
same (batch, 5, 512, 512) tensor and returns a (batch, 3, 512, 512) tensor.

Usage:
    python inference_onnx.py \
        --model  model_files/unet.onnx \
        --image  path/to/face.jpg \
        --source_age 25 \
        --target_age 70 \
        --output aged.png

Dependencies (in addition to the project's existing requirements):
    pip install onnxruntime          # CPU
    pip install onnxruntime-gpu      # GPU (CUDA)
"""

import argparse
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(".")


# ---------------------------------------------------------------------------
# ONNX session wrapper
# ---------------------------------------------------------------------------

class OnnxUNet:
    """Wraps an ONNX Runtime session so it can replace the PyTorch UNet."""

    def __init__(self, onnx_path: str, use_gpu: bool = False):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Expose a .device so helper code can query it if needed
        self.device = torch.device("cuda" if use_gpu else "cpu")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.  x: (B,5,512,512) tensor → (B,3,512,512) tensor."""
        x_np = x.cpu().numpy().astype(np.float32)
        out_np = self.session.run([self.output_name], {self.input_name: x_np})[0]
        return torch.from_numpy(out_np).to(self.device)


# ---------------------------------------------------------------------------
# Sliding-window inference (mirrors scripts/test_functions.py)
# ---------------------------------------------------------------------------

def _load_mask(path: str, size: int) -> torch.Tensor:
    try:
        return torch.from_numpy(np.array(Image.open(path).convert("L"))) / 255.0
    except FileNotFoundError:
        # If mask assets are missing, use an all-ones mask as a fallback
        return torch.ones(size, size)


def sliding_window_onnx(
    model: OnnxUNet,
    input_tensor: torch.Tensor,
    window_size: int = 512,
    stride: int = 256,
    mask_path: str = "assets/mask1024.jpg",
    small_mask_path: str = "assets/mask512.jpg",
) -> torch.Tensor:
    """Apply the UNet via a sliding window and return the blended result."""

    n, c, h, w = input_tensor.shape
    mask       = _load_mask(mask_path, max(h, w))[:h, :w]
    small_mask = _load_mask(small_mask_path, window_size)

    output_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32)
    count_tensor  = torch.zeros((n, 3, h, w), dtype=torch.float32)

    add = 2 if window_size % stride != 0 else 1

    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y : y + window_size, x : x + window_size]

            with torch.no_grad():
                out = model(window).cpu()

            output_tensor[:, :, y : y + window_size, x : x + window_size] += out * small_mask
            count_tensor [:, :, y : y + window_size, x : x + window_size] += small_mask

    count_tensor = torch.clamp(count_tensor, min=1.0)
    output_tensor /= count_tensor
    output_tensor *= mask
    return output_tensor


# ---------------------------------------------------------------------------
# Full image pipeline
# ---------------------------------------------------------------------------

def process_image_onnx(
    model: OnnxUNet,
    image: Image.Image,
    source_age: int,
    target_age: int,
    window_size: int = 512,
    stride: int = 256,
) -> Image.Image:
    """Age a face image from source_age to target_age using the ONNX model."""

    try:
        import face_recognition
        np_image = np.array(image)
        fl = face_recognition.face_locations(np_image)[0]

        margin_y_t = int((fl[2] - fl[0]) * 0.63 * 0.85)
        margin_y_b = int((fl[2] - fl[0]) * 0.37 * 0.85)
        margin_x   = int((fl[1] - fl[3]) // (2 / 0.85))
        margin_y_t += 2 * margin_x - margin_y_t - margin_y_b

        l_y = max(fl[0] - margin_y_t, 0)
        r_y = min(fl[2] + margin_y_b, np_image.shape[0])
        l_x = max(fl[3] - margin_x, 0)
        r_x = min(fl[1] + margin_x, np_image.shape[1])

        cropped = np_image[l_y:r_y, l_x:r_x]
    except Exception:
        # Fall back to full image if face detection fails
        np_image = np.array(image)
        l_y, r_y, l_x, r_x = 0, np_image.shape[0], 0, np_image.shape[1]
        cropped = np_image

    orig_size = cropped.shape[:2]       # (H, W)
    input_size = (1024, 1024)

    img_tensor   = transforms.ToTensor()(Image.fromarray(np.array(image)))
    crop_tensor  = transforms.ToTensor()(Image.fromarray(cropped))
    crop_resized = transforms.Resize(input_size, interpolation=Image.BILINEAR, antialias=True)(crop_tensor)

    src_ch = torch.full_like(crop_resized[:1], source_age / 100.0)
    tgt_ch = torch.full_like(crop_resized[:1], target_age / 100.0)
    inp    = torch.cat([crop_resized, src_ch, tgt_ch], dim=0).unsqueeze(0)  # (1,5,1024,1024)

    aged_crop = sliding_window_onnx(model, inp, window_size, stride)        # (1,3,1024,1024)

    aged_crop_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(
        aged_crop
    )

    img_tensor[:, l_y:r_y, l_x:r_x] += aged_crop_resized.squeeze(0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    return transforms.functional.to_pil_image(img_tensor)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Face re-aging via ONNX Runtime")
    parser.add_argument("--model",      required=True,  help="Path to unet.onnx")
    parser.add_argument("--image",      required=True,  help="Input image path")
    parser.add_argument("--source_age", type=int, required=True)
    parser.add_argument("--target_age", type=int, required=True)
    parser.add_argument("--output",     default="aged_output.png")
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride",      type=int, default=256)
    parser.add_argument("--gpu",  action="store_true", help="Use CUDA execution provider")
    args = parser.parse_args()

    print(f"Loading ONNX model from {args.model} ...")
    model = OnnxUNet(args.model, use_gpu=args.gpu)

    image = Image.open(args.image).convert("RGB")
    print(f"Processing image (source_age={args.source_age}, target_age={args.target_age}) ...")
    result = process_image_onnx(model, image, args.source_age, args.target_age,
                                 args.window_size, args.stride)

    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

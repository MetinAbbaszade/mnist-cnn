"""Turn a real photo of a handwritten digit into an MNIST-style input tensor.

MNIST digits are white on black, centered, fit in a 20x20 box inside a 28x28 image,
and the model was trained on pixel values scaled to 0-1. Every step below moves a
phone photo closer to that format, and each intermediate image is returned so the
preprocessing can be shown explicitly.
"""
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


def preprocess(path):
    steps = {}

    # 1. Open the photo and fix phone rotation (EXIF orientation)
    img = ImageOps.exif_transpose(Image.open(path))
    steps["1. original"] = np.array(img.convert("RGB"))

    # 2. Grayscale: 3 color channels -> 1
    gray = img.convert("L")
    steps["2. grayscale"] = np.array(gray)

    # 3. Invert relative to the local background.
    #    A heavy blur keeps only the lighting of the paper (shadows included).
    #    Ink is darker than its own surroundings, a shadow is not, so subtracting
    #    removes shadows and turns dark ink into bright strokes (white on black).
    radius = max(gray.size) // 20
    background = np.array(gray.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
    ink = np.clip(background - np.array(gray).astype(np.float32), 0, None)
    steps["3. inverted (shadow removed)"] = ink

    # 4. Clean: keep strong strokes, then keep only the large connected blobs
    #    (removes grid lines, specks and dust; small parts of the digit are kept
    #    as long as they are at least 20% of the largest blob)
    mask = ink > ink.max() * 0.4
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= sizes.max() * 0.2]
        mask = np.isin(labels, keep)
    cleaned = np.where(mask, ink, 0)
    cleaned = cleaned / cleaned.max() * 255

    # 5. Crop to the digit's bounding box
    ys, xs = np.nonzero(cleaned)
    cropped = cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    steps["4. cleaned + cropped"] = cropped

    # 6. Resize so the longer side is 20 px (keeping the shape), center on a 28x28 black canvas
    h, w = cropped.shape
    scale = 20 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - new_w) // 2, (28 - new_h) // 2))
    final = np.array(canvas)
    steps["5. final 28x28"] = final

    # 7. Normalize to 0-1 and shape [1, 1, 28, 28], exactly like the training data
    x = torch.tensor(final, dtype=torch.float32) / 255.0
    x = x.unsqueeze(0).unsqueeze(0)
    return steps, x

"""Load the trained CNN from disk and classify real handwritten digit photos.

No training happens here: the weights are read from models/cnn_mnist.pt into a
freshly built network. This proves the model works without retraining.

Usage:
    python predict.py my_digits/*.jpeg
    python predict.py my_digits/*.jpeg --plot images/custom_digits.png

Name each photo starting with its true digit (e.g. 7.jpeg, 7_crossed.jpeg)
so the script can mark predictions as correct or wrong.
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch

from model import build_cnn
from preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digits with the trained CNN.")
    parser.add_argument("images", nargs="+", help="paths to digit photos")
    parser.add_argument("--weights", default="models/cnn_mnist.pt", help="saved model weights")
    parser.add_argument("--plot", default=None, help="save a figure of every preprocessing step to this path")
    args = parser.parse_args()

    model = build_cnn()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    results = []
    correct = labeled = 0
    for path in sorted(args.images):
        steps, x = preprocess(path)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred = probs.argmax().item()

        name = os.path.basename(path)
        true = name[0] if name[0].isdigit() else None
        if true is None:
            mark = ""
        else:
            labeled += 1
            correct += str(pred) == true
            mark = "correct" if str(pred) == true else "WRONG"
        print(f"{name:<28} true {true or '?'}  pred {pred}  ({probs[pred]:.1%})  {mark}")
        results.append((name, steps, probs, pred, true))

    if labeled:
        print(f"\n{correct}/{labeled} correct")

    if args.plot:
        fig, axes = plt.subplots(len(results), 6, figsize=(14, 2.4 * len(results)), squeeze=False)
        for row, (name, steps, probs, pred, true) in zip(axes, results):
            for ax, (title, image) in zip(row, steps.items()):
                ax.imshow(image, cmap="gray")
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            row[5].bar(range(10), probs.numpy())
            row[5].set_xticks(range(10))
            row[5].set_ylim(0, 1)
            row[5].set_title(f"true {true or '?'}, pred {pred} ({probs[pred]:.0%})", fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plt.savefig(args.plot, dpi=120)
        print(f"Saved figure to {args.plot}")


if __name__ == "__main__":
    main()

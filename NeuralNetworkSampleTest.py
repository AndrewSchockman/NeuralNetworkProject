
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import load
from torchvision import transforms
from torchvision.transforms import ToTensor

# ---- Import a model object named `clf` from training file ----

from NeuralNetwork import clf


# ---------------- User knobs ----------------
numberOfTrials = 1000  # random samples to test

# -------------- Preprocessing helpers --------------
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
_NORMALIZE = transforms.Normalize(MNIST_MEAN, MNIST_STD)

def removeTransparency(img_rgba: Image.Image) -> Image.Image:
    #Replace transparent background with white; return grayscale 'L' image.
    white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    white_bg.paste(img_rgba, (0, 0), img_rgba)
    return white_bg.convert("L")

def getImageState(img: Image.Image) -> str:
    #Return 'T' if any border pixel is transparent; else 'O' (opaque).
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    coords = [(0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,0),(0,h//2),(w-1,h//2),(w//2,h-1)]
    for x, y in coords:
        if img.getpixel((x, y))[3] < 255:
            return "T"
    return "O"

def _center_resize_28(imgL: Image.Image, *, bright_foreground: bool) -> Image.Image:
    
    #Crop to digit box, add small padding, scale largest side to 20px, center on 28x28.
    #bright_foreground=True means white digit on black; False means black digit on white.
    
    arr = np.asarray(imgL, dtype=np.uint8)

    # After autocontrast, background is near 0 (wob) or 255 (bow). Keep ALL ink.
    mask = (arr > 0) if bright_foreground else (arr < 255)
    if not mask.any():
        return imgL.resize((28, 28))  # fallback

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    y1, y2 = np.where(rows)[0][0], np.where(rows)[0][-1] + 1
    x1, x2 = np.where(cols)[0][0], np.where(cols)[0][-1] + 1
    crop = imgL.crop((x1, y1, x2, y2))

    # Gentle padding avoids shaving tips (helps 6 tails, 7 top bars)
    pad_fill = 0 if bright_foreground else 255
    crop = ImageOps.expand(crop, border=2, fill=pad_fill)

    # Scale & center
    w, h = crop.size
    scale = 20 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    crop = crop.resize((new_w, new_h), Image.BILINEAR)

    canvas_bg = 0 if bright_foreground else 255
    canvas = Image.new("L", (28, 28), canvas_bg)
    canvas.paste(crop, ((28 - new_w) // 2, (28 - new_h) // 2))
    return canvas

def _border_mean(imgL: Image.Image) -> float:
    #Mean of outermost border pixels (post-centering).
    a = np.asarray(imgL, dtype=np.uint8)
    border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    return float(border.mean())

def _to_tensor(imgL: Image.Image) -> torch.Tensor:
    t = ToTensor()(imgL)           # (1,28,28) in [0,1]
    t = _NORMALIZE(t)
    return t.unsqueeze(0)          # (1,1,28,28)

def preprocess_choose(ImagePath: Path) -> torch.Tensor:
    
    #Build two candidates (white-on-black & black-on-white), both centered.
    #Decide polarity by border brightness AFTER centering (robust), then return tensor.
    
    rgba = Image.open(ImagePath).convert("RGBA")
    baseL = removeTransparency(rgba) if getImageState(rgba) == "T" else rgba.convert("L")

    # Candidate A: white digit on black (MNIST-like)
    wob = ImageOps.invert(baseL)
    wob = ImageOps.autocontrast(wob)
    wob = _center_resize_28(wob, bright_foreground=True)

    # Candidate B: black digit on white
    bow = ImageOps.autocontrast(baseL)
    bow = _center_resize_28(bow, bright_foreground=False)

    # Decide polarity via border mean (avoid overconfident logits on wrong polarity)
    choose_wob = (_border_mean(wob) < 64) or (_border_mean(bow) > 192)
    chosen = wob if choose_wob else bow

    return _to_tensor(chosen)

# -------------- Main: random-sample test --------------
if __name__ == "__main__":
    # Load weights and prep model
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
    clf.eval()
    device = next(clf.parameters()).device

    PROJECT_ROOT = Path(__file__).resolve().parent
    indSuccess = [0] * 10
    indTrials  = [0] * 10
    success = failure = 0

    for _ in range(numberOfTrials):
        digit = random.randrange(0, 10)
        archiveNum = random.randrange(0, 10773)
        print(f"Expected digit- {digit} Archive Number- {archiveNum}")

        IMG_PATH = PROJECT_ROOT / "archive" / "dataset" / str(digit) / str(digit) / f"{archiveNum}.png"
        print(f"Opening file path: {IMG_PATH}")
        if not IMG_PATH.exists():
            print("File missing; skipping.\n")
            continue

        with torch.no_grad():
            x = preprocess_choose(IMG_PATH).to(device)
            logits = clf(x)
            predicted_digit = logits.argmax(dim=-1).item()

        print(f"Observed digit:{predicted_digit}")
        print(f"Expected digit:{digit}")

        indTrials[digit] += 1
        if predicted_digit == digit:
            success += 1
            indSuccess[digit] += 1
            print("Success\n")
        else:
            failure += 1
            print("Failure\n")

    total_trials = sum(indTrials)
    print(f"Total successes: {success}")
    print(f"Total failures: {failure}")
    if total_trials:
        print(f"Overall accuracy: {success/total_trials*100:.2f}%  ({success}/{total_trials})\n")
    else:
        print("No valid trials ran (files missing?).\n")

    for i in range(10):
        trials = indTrials[i]
        ratio = (indSuccess[i] / trials) if trials else 0.0
        print(f"Number: {i}   Success/Trials: {indSuccess[i]}/{trials}   Ratio: {ratio:.4f}")

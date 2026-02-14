from PIL import Image
import numpy as np

def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_image_rgb(path: str, img_rgb: np.ndarray) -> None:
    Image.fromarray(img_rgb).save(path)

\
import cv2
from PIL import Image
import numpy as np

# Placeholder for potential future utility functions
# e.g., standardized image loading, resizing, preprocessing

def load_image_cv(file_path: str):
    """Loads an image using OpenCV."""
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image with OpenCV: {file_path}")
    return img

def load_image_pil(file_path: str):
    """Loads an image using Pillow."""
    try:
        img = Image.open(file_path)
        return img
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not load image with Pillow: {file_path}")
    except Exception as e:
        raise IOError(f"Error opening image {file_path} with Pillow: {e}")


"""
Image utility functions for the Vision2Voice pipeline.
"""
import logging
import os
from pathlib import Path
from typing import Union
import cv2
import numpy as np
logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def validate_image_path(image_path):
    path = Path(image_path).resolve()
    if not path.exists(): raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS: raise ValueError(f"Unsupported: {path.suffix}")
    return path

def load_image_rgb(image_path):
    path = validate_image_path(image_path)
    bgr = cv2.imread(str(path))
    if bgr is None: raise IOError(f"OpenCV could not decode: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def resize_and_pad(img, target_size=(224,224), pad_colour=(0,0,0)):
    h,w = img.shape[:2]
    tw,th = target_size
    scale = min(tw/w,th/h)
    new_w,new_h = int(w*scale)int(h*scale)
    resized = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_AREA)
    canvas = np.full((th,tw,3),pad_colour,dtype=np.uint8)
    xo,yo = (tw-new_w)//2,(th-new_h)//2
    canvas[yo:yo+new_h,xo:xo+new_w]=resized
    return canvas

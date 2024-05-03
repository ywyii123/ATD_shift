from calendar import c
import PIL
import cv2
from matplotlib import scale
import numpy as np
import torch

def center_crop(image, size):
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        cropped_image = image.crop((left, top, left + size, top + size))
        return cropped_image

if __name__ == "__main__":
    x0 = PIL.Image.open('/data/datasets/ImageNet/data/ImageNet2012/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG')
    x0 = center_crop(x0, 256)
    h, w = x0.size
    scale_factor = 0.25
    x1 = x0.resize((int(h * scale_factor), int(w * scale_factor)), resample = PIL.Image.BICUBIC)
    x1.save('./lq/1.JPEG')

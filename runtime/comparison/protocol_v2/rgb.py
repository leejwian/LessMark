"""The registered RGB source transform used by experiment 2."""
import numpy as np
import torch
from PIL import Image



def center_crop(image,size=256):
    # Same BOX/downsample + BICUBIC/center crop as official image_datasets.py.
    while min(image.size)>=2*size:
        image = image.resize(tuple(x//2 for x in image.size),resample=Image.Resampling.BOX)
    scale = size/min(image.size)
    image = image.resize(tuple(round(x*scale) for x in image.size),resample=Image.Resampling.BICUBIC)
    a = np.array(image.convert("RGB"))
    y,x = (a.shape[0]-size)//2,(a.shape[1]-size)//2
    return torch.from_numpy((a[y:y+size,x:x+size].astype(np.float32)/127.5-1).transpose(2,0,1).copy())

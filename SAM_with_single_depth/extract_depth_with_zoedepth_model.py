import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
   
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)

repo = "isl-org/ZoeDepth"
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_k.to(DEVICE)

image = Image.open("./Grounded-Segment-Anything/assets/road.png").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

save_path = './test_image_road.png'
save_raw_16bit(depth_tensor, save_path)


save_path = './test_image_road_color.png'

Image.fromarray(colored_depth).save(save_path)

import torch
import numpy as np
import deepinv as dinv
import tqdm
import os

from dataset import load_dataset
from models import ModuloSEFLnet

import matplotlib.pyplot as plt

MAX_VALUE = 4.0
THRESHOLD = 1.0
MODE = "floor"
n_channels = 3
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



physics = dinv.physics.SpatialUnwrapping(threshold=THRESHOLD, mode=MODE).to(device)
# model =  ModuloSEFLnet(mx=THRESHOLD, in_channels=n_channels, out_channels=n_channels).to(device)
# model.load_state_dict(torch.load(os.path.join("ckpts", "ModuloSEFLnet.pth")))
# # print numeber of parameters in K
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters in the model: {num_params/1e3:.2f} K")


psnr_fn = dinv.loss.metric.PSNR(max_pixel=MAX_VALUE)
ssim_fn = dinv.loss.metric.SSIM(max_pixel=MAX_VALUE)

# model.eval()


data_path = os.path.join("data", "unmodnet_test", "source")
img_path = os.listdir(data_path)[93]
print(img_path)

img = np.load(os.path.join(data_path, img_path))
img = torch.from_numpy(img).float().permute(2,0,1)  # HWC to CHW
img = img.unsqueeze(0).to(device)
img = img.clamp(min=0.0) * MAX_VALUE / img.max()

y   = physics(img)

# with torch.no_grad():
#     x_rec = model(y)
#     psnr = psnr_fn(x_rec, img).item()
#     ssim = ssim_fn(x_rec, img).item()
#     print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

psnr = 0
ssim = 0
x_rec = img.clone() # 
# plot results

x_rec /= MAX_VALUE
img /= MAX_VALUE

x_rec = x_rec.clamp(0.0, 1.0)


fig, axs = plt.subplots(1,3, figsize=(12,4))
axs[0].imshow(y[0].cpu().permute(1,2,0))
axs[0].set_title("Modulo Measurements")
axs[1].imshow(x_rec[0].cpu().permute(1,2,0))
axs[1].set_title(f"Reconstruction\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
axs[2].imshow(img[0].cpu().permute(1,2,0))
axs[2].set_title("Ground Truth")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()




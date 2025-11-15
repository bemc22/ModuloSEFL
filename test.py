import torch
import deepinv as dinv
import tqdm
import os

from dataset import load_dataset
from models import ModuloSEFLnet

import matplotlib.pyplot as plt

DATA_ROOT = os.path.join(".", "data", "unmodnet")
MAX_VALUE = 4.0
THRESHOLD = 1.0
MODE = "floor"
n_channels = 3
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_, test_dataset = load_dataset(DATA_ROOT, max_val=MAX_VALUE)


physics = dinv.physics.SpatialUnwrapping(threshold=THRESHOLD, mode=MODE).to(device)
model =  ModuloSEFLnet(mx=THRESHOLD, in_channels=n_channels, out_channels=n_channels).to(device)
model.load_state_dict(torch.load(os.path.join("ckpts", "ModuloSEFLnet.pth")))
# print numeber of parameters in K
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in the model: {num_params/1e3:.2f} K")


psnr_fn = dinv.loss.metric.PSNR(max_pixel=MAX_VALUE)
ssim_fn = dinv.loss.metric.SSIM(max_pixel=MAX_VALUE)

model.eval()

sample =  test_dataset[0]
img, _ = sample
img = img.unsqueeze(0).to(device)
y   = physics(img)

with torch.no_grad():
    x_rec = model(y)
    psnr = psnr_fn(x_rec, img).item()
    ssim = ssim_fn(x_rec, img).item()
    print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

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




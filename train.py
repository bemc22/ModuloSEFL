import torch
import deepinv as dinv
import tqdm
import os

from dataset import load_dataset
from models import ModuloSEFLnet


def scale_eq_loss(physics, x, model, loss_fn=None, num_trans=3, alpha=0.1):
    # scale equivariance loss
    total_loss = 0.0
    sat_range = 0.4

    for i in range(num_trans):
        # Apply a random scale transformation to x
        scale_factor = torch.rand(1).item() * sat_range - (sat_range / 2) + 1.0
        x_scale = x * scale_factor
        y_virtual = physics(x_scale)
        x_hat = model(y_virtual)
        loss = loss_fn(x_scale, x_hat)
        total_loss += loss

    w = alpha / num_trans

    return total_loss * w


DATA_ROOT = os.path.join(".", "data", "unmodnet")
MAX_VALUE = 4.0
THRESHOLD = 1.0
MODE = "floor"
n_channels = 3
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset, test_dataset = load_dataset(DATA_ROOT, max_val=MAX_VALUE)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

physics = dinv.physics.SpatialUnwrapping(threshold=THRESHOLD, mode=MODE).to(device)
model = ModuloSEFLnet(mx=THRESHOLD, in_channels=n_channels, out_channels=n_channels).to(
    device
)
fn_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# print numeber of parameters in K
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in the model: {num_params/1e3:.2f} K")


psnr_fn = dinv.loss.metric.PSNR(max_pixel=MAX_VALUE)
ssim_fn = dinv.loss.metric.SSIM(max_pixel=MAX_VALUE)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm.tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"
        ) as pbar:
            for batch in train_loader:

                x = batch[0].to(device)
                y = physics(x)

                optimizer.zero_grad()
                x_rec = model(y)
                loss = fn_loss(x_rec, x) + scale_eq_loss(physics, x, model, loss_fn=fn_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item():.6f}", avg=f"{(epoch_loss / pbar.n):.6f}"
                )

        # save model checkpoint
        save_path = os.path.join("ckpts", "ModuloSEFLnet.pth")
        torch.save(model.state_dict(), save_path)

        model.eval()
        with torch.no_grad():

            total_psnr = 0.0
            total_ssim = 0.0
            for batch in test_loader:
                x = batch[0].to(device)
                y = physics(x)
                x_rec = model(y)

                total_psnr += psnr_fn(x_rec, x).mean().item()
                total_ssim += ssim_fn(x_rec, x).mean().item()

            avg_psnr = total_psnr / len(test_loader)
            avg_ssim = total_ssim / len(test_loader)

            print(f"Test PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

# Scale Equivariance Regularization and Feature Lifting in High Dynamic Range Modulo Imaging
[Brayan Monroy](https://bemc22.github.io), [Jorge Bacca](https://scholar.google.com/citations?user=I5f1HjEAAAAJ&hl=es)

## Summary

This repository implements **ModuloSEFLnet**, a deep learning model for High Dynamic Range (HDR) imaging reconstruction from modulo measurements. The approach combines:

- **Feature Lifting**: Extracts gradient features from modulo measurements to enhance reconstruction
- **Scale Equivariance Regularization**: Enforces scale-invariant properties during training for robust HDR recovery
- **DRUNet Architecture**: Leverages a U-Net backbone with residual blocks for image reconstruction

The model tackles the inverse problem of recovering HDR images from modulo wrapping artifacts caused by sensor saturation at a threshold value.

### Key Components

- **Physics Model**: Spatial unwrapping with configurable threshold (default: 1.0) and floor mode
- **Network**: Modified DRUNet with feature lifting (gradient-based) input processing
- **Training Loss**: MSE loss + scale equivariance regularization
- **Metrics**: For the demo we report PSNR and SSIM as baseline metrics; however, we recommend using HDR-specific perceptual metrics for a more accurate assessment of HDR image quality:
    - HDR-VDP: perceptual difference metric designed for HDR content (compute on linear HDR radiance or converted to scene-referred units).
    - PU21-based metrics: convert HDR to the PU21 perceptually-uniform domain and report PU-PSNR / PU-SSIM for more reliable fidelity measures.

## Requirements

This project requires the **[deepinv](https://github.com/deepinv/deepinv)** library for physics modeling, network architectures, and metrics.

Install dependencies:
```bash
pip install deepinv torch torchvision matplotlib numpy tqdm
```

## Usage

### Training

Train the ModuloSEFLnet model on the UnModNet HDR dataset:

```bash
python train.py
```
The training script automatically evaluates on the test set after each epoch and reports PSNR/SSIM metrics.

### Testing

Test the trained model and visualize results:

```bash
python test.py
```
This loads the trained checkpoint from `./ckpts/ModuloSEFLnet.pth`.

## Dataset Structure

Expected dataset format:
```
data/
  unmodnet/
    source/
      *.npy files (training images)
  unmodnet_test/
    source/
      *.npy files (test images)
```

Images should be stored as `.npy` files and will be normalized to the range `[0, MAX_VALUE]`.

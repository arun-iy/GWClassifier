# GWClassifier — Gravitational Wave Signal Classification

Binary classification of astrophysical transient signals (gravitational waves vs. noise glitches)
using a convolutional neural network trained on LIGO O3 data.

## Results
| Metric | Value |
|--------|-------|
| Baseline accuracy | ~75% |
| Final accuracy | 87.3% (+12.3%) |
| Dataset | LIGO O3 public dataset |

## Method
- CNN architecture: 1D Inception-based CNN
- Preprocessing: bandpass filtering, whitening, Q-transform spectrograms
- Training: MixUp data augmentation, Stochastic Weight Averaging (SWA)

## Model Architecture
The classifier is a **1D Inception-based CNN** designed for multivariate time-series signals,
taking simultaneous input from both LIGO detectors (Hanford H1 and Livingston L1).

### Input
The input tensor has shape `[batch, 2, 4096]`, where the two channels correspond to the
whitened, bandpass-filtered strain time-series from H1 and L1. Processing both detectors
as input channels enables cross-detector pattern learning — critical for distinguishing
real astrophysical events from single-detector glitches.

### Inception Blocks
The network stacks **four Inception blocks**, each with five parallel branches operating
over the same input with different receptive fields:

| Branch | Kernel | Captures |
|--------|--------|----------|
| Branch 1 | k=1 | Pointwise / channel mixing |
| Branch 2 | k=3 | Fine temporal structure |
| Branch 3 | k=5 | Medium-scale features |
| Branch 4 | k=7 | Long-range dependencies |
| Branch 5 | MaxPool(3) + Conv(1) | Aggregated context |

Branch outputs are **concatenated along the channel axis**, then passed through
BatchNorm1d and ReLU. Each block downsamples by stride=2, progressively compressing
the time dimension while expanding feature representation.

| Block | Input channels | Filters/branch | Output channels | Stride |
|-------|---------------|----------------|-----------------|--------|
| 1     | 2             | 32             | 160             | 2      |
| 2     | 160           | 64             | 320             | 2      |
| 3     | 320           | 64             | 320             | 2      |
| 4     | 320           | 64             | 320             | 2      |

Dropout (p=0.4) is applied after Block 2 to regularize early feature maps.

### Classification Head
After Block 4, **global average pooling and global max pooling** are applied in parallel
and concatenated, yielding a 640-dim feature vector that captures both mean activation
and peak response across the time dimension.

## Stack
Python · PyTorch · NumPy · Matplotlib · SciPy

## How to run
\`\`\`bash
pip install -r requirements.txt
jupyter notebook notebooks/GWClassifier.ipynb
\`\`\`

# StyleGAN-like Generator on Fashion-MNIST in PyTorch

This repository contains a lightweight, PyTorch-based implementation of a StyleGAN-inspired generative model. It includes a multi-layer Mapping Network, a Synthesis Network equipped with Adaptive Instance Normalization (AdaIN), style mixing, and latent space linear interpolation.

---

## 🛠️ Libraries Used

The project is built using Python 3 and the following primary libraries:
- **PyTorch (`torch`)**: Core deep learning framework used to design and train the generator and discriminator neural networks.
- **Torchvision (`torchvision`)**: Used for downloading, loading, and transforming the Fashion-MNIST dataset, as well as utility functions like `make_grid` for plotting.
- **NumPy (`numpy`)**: Used for handling numerical calculations, array transformations, and generating values for latent interpolation.
- **Matplotlib (`matplotlib.pyplot`)**: Used to visualize style mixing outputs and latent space interpolation grids.

---

## 🧩 Architecture Components

### 1. Mapping Network
- **Input**: Random latent noise vector $z \in \mathbb{R}^{128}$ (sampled from a standard normal distribution).
- **Structure**: 8-layer fully connected (MLP) network with ReLU activations.
- **Role**: Normalizes $z$ (PixelNorm-like) and maps it to an intermediate style space $w \in \mathbb{R}^{128}$. This unentangles the latent space, making features easier to control.

### 2. Synthesis Network with AdaIN
- **Constant Input**: A learned $4 \times 4 \times 128$ tensor serves as the starting point of generation.
- **Adaptive Instance Normalization (AdaIN)**: Aligns the channel-wise mean and variance of the feature maps with the style scaling parameters ($y_s$) and bias parameters ($y_b$) generated from $w$ via linear projections.
- **Resolution Scaling blocks**: Progresses from $4 \times 4 \rightarrow 8 \times 8 \rightarrow 16 \times 16 \rightarrow 32 \times 32$ using bilinear upsampling layers followed by convolutional layers, AdaIN normalization, and ReLU activations.

### 3. Discriminator
- **Structure**: A standard Deep Convolutional GAN (DCGAN) style classifier using 2D convolutions, batch normalization, and LeakyReLU activations.
- **Role**: Classifies images as either real (from Fashion-MNIST dataset) or fake (produced by the Generator).

---

## 🔄 Step-by-Step Procedure

### Step 1: Dataset Setup & Preprocessing
- Download the Fashion-MNIST dataset.
- Resize images from $28 \times 28$ to $32 \times 32$.
- Normalize pixel values to the range $[-1, 1]$.
- Convert single-channel grayscale images to 3-channel (RGB) replicas to match standard StyleGAN channel dimensions.

### Step 2: Adversarial Training Loop
- Models are trained using the **Non-Saturating GAN Loss** with BCE loss functions.
- **Discriminator step**: Train to minimize classification error by maximizing $D(x)$ for real images and minimizing $D(G(z))$ for generated images.
- **Generator step**: Train to maximize the probability of the Discriminator classifying fake images as real ($D(G(z)) \approx 1$).
- Optimization is performed using Adam with a learning rate of `0.0002` and custom momentum betas of `(0.0, 0.99)`.

### Step 3: Style Mixing Verification
- Run model evaluation using two different source styles, Style A ($z_1$) and Style B ($z_2$).
- Mix styles by passing Style A into the early layers of the Synthesis network (e.g., blocks 1 and 2) and injecting Style B into the later blocks (blocks 3 and 4).

### Step 4: Latent Interpolation Verification
- Linearly interpolate between two randomly sampled latent vectors $z_{start}$ and $z_{end}$ using a weight coefficient $\alpha \in [0, 1]$.
- Generate images for intermediate vectors to verify the continuous smoothness of the latent space.

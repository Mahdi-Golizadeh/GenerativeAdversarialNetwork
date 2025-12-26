# GenerativeAdversarialNetwork

This repository contains clean, well-commented Jupyter notebooks implementing key Generative Adversarial Network (GAN) variants — **Vanilla GAN**, **DCGAN**, and **WGAN-GP** — trained on the **MNIST** dataset. Each notebook demonstrates architecture design, training procedure, loss interpretation (including expected behavior of negative critic/generator losses in WGAN-GP), and qualitative results.

> ✅ **Note**: Includes practical insights on WGAN-GP training — *negative critic values are normal* and reflect the unbounded nature of the Wasserstein loss.

---

## 📁 Notebooks

| File | Description |
|------|-------------|
| [`Vanilla_GAN_MNIST.ipynb`](./Vanilla_GAN_MNIST.ipynb) | Basic GAN with fully-connected generator and discriminator. Illustrates fundamental GAN concepts and common challenges (e.g., mode collapse, vanishing gradients). |
| [`DCGAN_MNIST.ipynb`](./DCGAN_MNIST.ipynb) | Deep Convolutional GAN (DCGAN) using convolutional layers, batch normalization, and LeakyReLU. Improved training stability and sample quality over Vanilla GAN. |
| [`WGAN_GP_MNIST.ipynb`](./WGAN_GP_MNIST.ipynb) | Wasserstein GAN with Gradient Penalty (WGAN-GP). Replaces sigmoid output and BCE loss with linear critic and Wasserstein loss + gradient penalty. More stable training, meaningful loss curves, and interpretable critic scores (can be negative — expected). |

All notebooks use **PyTorch** and include:
- Reproducible seeds
- Training loop with logging
- Generated sample visualization every few epochs
- Tips for hyperparameter tuning

---

## 🛠️ Setup

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- numpy, matplotlib, tqdm, jupyter

### Installation
```bash
git clone https://github.com/Mahdi-Golizadeh/GenerativeAdversarialNetwork.git
cd GenerativeAdversarialNetwork
pip install torch torchvision numpy matplotlib jupyter tqdm

# Adversarial Attacks on Convolutional Neural Networks (CNNs)

This repository contains implementations and evaluations of adversarial threats to Deep Learning models, specifically focusing on **Evasion Attacks** and **Model Inversion Attacks**. This work was developed as part of the *AI60006: Dependable and Secure AI-ML* course.

##  Overview

The project explores the vulnerability of various architectures (MLPs, CNNs, and VGG19) using the **Adversarial Robustness Toolbox (ART)**. It includes practical demonstrations of how small, often invisible perturbations can fool a classifier, and how model parameters can be exploited to reconstruct private training data.

### Key Features
* **Evasion Attacks:** Benchmarking FGSM and PGD attacks across MNIST, CIFAR-10, and ImageNet.
* **Model Inversion:** Reconstructing face images from the AT&T Face Dataset and object classes from CIFAR-10 using the MI-Face algorithm.
* **Interactive Visualization:** A Streamlit dashboard to visualize internal CNN feature maps and activations.

---

##  Repository Structure

| File | Description |
| :--- | :--- |
| `evasion_attack.ipynb` | Implementation of Random Noise, $l_{\infty}$, and $l_2$ attacks on VGG19, custom MLP, and CNNs. |
| `inversion_atnt.ipynb` | Privacy attack to reconstruct individual faces from a model trained on the AT&T dataset. |
| `inversion_cifar10.ipynb` | Class-level reconstruction of CIFAR-10 images using the MI-Face algorithm. |
| `atnt_app.py` | Streamlit application for real-time CNN feature visualization and layer-by-layer analysis. |

---

## Tech Stack

* **Deep Learning:** [PyTorch](https://pytorch.org/)
* **Adversarial Security:** [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
* **Deployment/Viz:** [Streamlit](https://streamlit.io/), Matplotlib, OpenCV
* **Datasets:** MNIST, CIFAR-10, AT&T Face Dataset, ImageNet

---

## Detailed Implementations

### 1. Evasion Attacks
We evaluate model robustness by generating adversarial examples that minimize the probability of the correct class:
* **Random Noise:** Baseline evaluation of robustness against stochastic perturbations.
* **Fast Gradient Sign Method (FGSM):** A one-step gradient-based attack ($l_{\infty}$ bounded).
* **Projected Gradient Descent (PGD):** An iterative, more powerful version of FGSM.

### 2. Model Inversion (Privacy)
Using the **MI-Face algorithm**, we treat the model as an oracle to reconstruct sensitive training data. 
* **Target:** A 6-layer CNN trained on the AT&T dataset ($40$ classes).
* **Outcome:** Successfully recovered recognizable facial features of specific subjects from the model weights.

### 3. Feature Visualizer (`atnt_app.py`)
An interactive tool to understand how CNNs process input:
* **Layer 1:** Shows low-level edge and texture detection (32 channels).
* **Layer 2:** Shows high-level feature aggregation (64 channels).
* **Predictions:** Real-time softmax probability distribution display.

---

## Setup & Usage

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Anirvan-Krishna/adversarial_attacks_on_cnns.git](https://github.com/Anirvan-Krishna/adversarial_attacks_on_cnns.git)
   cd adversarial_attacks_on_cnns

2. **Install Requirements**
   ```bash
   pip install adversarial-robustness-toolbox torch torchvision streamlit opencv-python matplotlib

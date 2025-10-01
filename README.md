# 🧠 CIFAR-10 Classification — Vision Transformers vs CNN

This repository contains three deep learning projects I implemented from scratch to compare different architectures for **image classification on the CIFAR-10 dataset**:

- 🧰 **Custom CNN** — A deep convolutional neural network with skip connections.  
- 🧠 **Vision Transformer (ViT)** — A patch-based transformer trained from scratch.  
- ⚡ **Compact Convolutional Transformer (CCT)** — A hybrid architecture combining convolutional tokenization with transformer sequence modeling.

All three models — CNN, ViT, and CCT — achieve **over 90% test accuracy** on CIFAR-10 after full training, demonstrating that carefully designed architectures and training strategies can close the gap between classical CNNs and transformer-based models on small datasets.


---

## 📂 Repository Structure

- **notebooks/**: Each notebook is fully self-contained, including data loading, model definition, training, evaluation, and plotting.  
- **plots/**: Training curves (accuracy and loss) exported after runs for quick comparison.

---

## 🧠 Architectures Overview

### 🧰 Custom CNN
A deep convolutional neural network with skip connections between blocks.  
This model represents the **classical baseline**: hierarchical feature extraction, local receptive fields, and efficient training.

- **Augmentations**: Random crop, horizontal flip  
- **Scheduler**: OneCycleLR  
- **Epochs**: 40  
- **Strengths**: Fast convergence, stable performance with few parameters.

---

### 🧠 Vision Transformer (ViT)
A transformer that treats image patches as tokens and processes them using self-attention, without any convolutions.

- **Patch embedding** + class token + multi-head self-attention  
- **Scheduler**: Linear warmup → Cosine annealing  
- **Epochs**: 250  
- **Augmentations**: Standard crop + flip  
- **Strengths**: Captures long-range dependencies; demonstrates transformer capacity on small datasets.  
- **Challenges**: Requires careful scheduling and longer training to stabilize.

---

### ⚡ Compact Convolutional Transformer (CCT)
A hybrid model that uses a **convolutional tokenizer** instead of patch embeddings and **sequence pooling** instead of a class token.  
This improves data efficiency and stability on small images like CIFAR-10.

- **Tokenizer**: Convolutional layers before transformer blocks  
- **Pooling**: Learnable sequence pooling instead of a class token  
- **Regularization**: Label smoothing, dropout, AdamW with weight decay  
- **Augmentations**: MixUp and CutMix with scheduled deactivation mid-training  
- **Scheduler**: Linear warmup → Cosine annealing  
- **Epochs**: 300  
- **Strengths**: Faster convergence than ViT, better inductive biases, fewer parameters.

---

## 🧪 Training Strategies

| Model   | Epochs | LR Schedule                | Regularization                          | Augmentations                  |
|--------|--------|----------------------------|------------------------------------------|--------------------------------|
| CNN    | 40     | OneCycleLR                 | Dropout                                 | Crop, Flip                     |
| ViT    | 250    | Linear Warmup + Cosine     | Dropout, Weight Decay                                 | Crop, Flip                     |
| CCT    | 300    | Linear Warmup + Cosine     | Label smoothing, Dropout, Weight Decay  | Crop, Flip, MixUp, CutMix      |

**MixUp & CutMix** are particularly effective for transformer models:  
They smooth the decision boundary and improve generalization, especially in early and mid stages of training.  
Later in training, these techniques are turned off to let the model focus on natural examples.

---

## 📊 Results & Insights

All models exceed **90% test accuracy**, with different convergence behaviors and efficiency trade-offs.

The figure below summarizes the training dynamics across all three models.

| Model | Convergence Speed | Stability | Final Performance | Parameter Efficiency |
|-------|--------------------|----------|--------------------|-----------------------|
| CNN   | ✅ Fast           | ✅ Stable | 🟡 Good baseline   | ✅ High              |
| ViT   | 🟡 Slow          | 🟡 Needs scheduling | 🟢 Strong after long training | 🟡 Moderate |
| CCT   | 🟢 Faster than ViT | 🟢 Stable | 🟢 Competitive     | 🟢 Fewer params than ViT |

- **CNN** is fast and reliable but lacks the flexibility of transformers.  
- **ViT** is powerful but data-hungry and requires more careful training on small datasets.  
- **CCT** combines the best of both worlds: good inductive biases, faster convergence, and competitive accuracy.

Plots for accuracy and loss over time are included in the `plots/` folder.

---
# 👤 Author
**Moneer Ghanem**  
Computer Science & Computational Neuroscience Student  
📫 [LinkedIn](https://www.linkedin.com/in/moneerghanem)  




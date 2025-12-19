# SaFF-AD

**SaFF-AD** extends the Self-adapting Forward-Forward Network (**[SaFF-Net](https://github.com/ividja/SaFF-Net)**) to **self-supervised anomaly detection** in medical imaging. The framework builds on the Forward-Forward Algorithm (FFA) and its convolutional variant (CFFA) to enable **resource-efficient, layer-wise training without back-propagation**, while supporting both **task-specific anomaly objectives** and **general representation learning**.

SaFF-AD is designed for settings in which computational resources, memory, or energy consumption are constrained, and where labelled anomalous data are scarce or unavailable. The framework supports 1D, 2D, and 3D medical imaging data and scales efficiently across resolutions and hardware configurations.

This repository contains the complete SaFF-AD pipeline, including self-configuration, Forward-Forward training, and anomaly inference.

**_NOTE: An updated implementation for SaFF-AD will be released soon. Until then, the repository contains the SaFF-Net codebase._**


---

## Key Features

- **Self-supervised anomaly detection** based on Forward-Forward learning  
- **Convolutional Forward-Forward Algorithm (CFFA)** for 2D and 3D medical images  
- **Anomaly-specific and general loss functions**
  - $\mathcal{L}_\text{anomaly}$ for direct anomaly separation
  - $\mathcal{L}_\text{general}$ for robust representation learning
- **Layer-wise optimisation without back-propagation**
- **Self-adapting configuration** based on data characteristics and hardware constraints
- **Highly parameter-efficient models**, competitive with back-propagation baselines
- Support for **one-shot and iterative training regimes**

---

## Forward-Forward Algorithm

![Forward Forward Algorithm](/figures/FF_algorithm.png)

The Forward-Forward Multi-Layer Perceptron and Forward-Forward Convolutional Neural Network follow the layer-wise optimisation scheme introduced by Hinton (2022). Positive and negative samples are generated via label encoding or synthetic anomaly construction and processed independently at each layer. After layer normalisation, the **orientation** of the activation is forwarded to the next layer, while its **magnitude** defines the layer goodness. Layers are trained to assign high goodness to positive samples and low goodness to negative samples. During inference, goodness scores are aggregated across layers to perform classification or anomaly detection.

---

## SaFF-AD Framework

![Framework](/figures/FF_framework.png)

The SaFF-AD framework performs **automatic self-configuration** based on training data properties (e.g. modality, dimensionality, resolution, intensity distribution) and available hardware resources (e.g. GPU memory). This process determines architectural parameters, optimisation settings, and training modes. The framework supports self-supervised and supervised training, pruning, early stopping, calibration, and post-processing, enabling flexible deployment across applications and compute environments.

---

## Efficiency

![Efficiency](/figures/FF_efficiency.png)

Comparison of classification and anomaly detection performance versus model size. SaFF-AD achieves competitive accuracy, AUC, and average precision with significantly fewer parameters than back-propagation-based MLPs and CNNs, particularly in large-batch and one-shot training scenarios.

---

## Publication

The original SaFF-Net framework was introduced in:

**Resource-efficient medical image analysis with self-adapting forward-forward networks**  
Johanna P. Müller, Bernhard Kainz  
MICCAI Workshop on Machine Learning in Medical Imaging (MLMI), 2024  

https://link.springer.com/chapter/10.1007/978-3-031-73290-4_18

SaFF-AD extends this work with a focus on **self-supervised anomaly detection** using Forward-Forward learning.

---

## Citation for SaFF-Net

```bibtex
@inproceedings{muller2024resource,
  title={Resource-efficient medical image analysis with self-adapting forward-forward networks},
  author={M{\"u}ller, Johanna P and Kainz, Bernhard},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={180--190},
  year={2024},
  organisation={Springer}
}

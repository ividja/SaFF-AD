# SaFF-Net

We introduce a fast Self-adapting Forward-Forward Network (**SaFF-Net**) for medical imaging analysis, mitigating power consumption and resource limitations, which currently primarily stem from the prevalent reliance on back-propagation for model training and fine-tuning. Building upon the recently proposed Forward-Forward Algorithm (FFA), we introduce the Convolutional Forward-Forward Algorithm (CFFA), a parameter-efficient reformulation that is suitable for advanced image analysis and overcomes the speed and generalization constraints of the original FFA. To address hyper-parameter sensitivity we are also introducing a self-adapting framework, which tunes **SaFF-Net** to the optimal settings.
Our approach enables more effective model training and eliminates the previously essential requirement for an arbitrarily chosen Goodness function in FFA.
We evaluate our approach on several benchmarking datasets in comparison with standard Back-Propagation (BP) neural networks showing that FFA-based networks can compete with standard models with notably fewer parameters and function evaluations, especially, in one-shot scenarios and large batch sizes. [1]

[Paper]

## Forward Forward Algorithm
![Forward Forward Algorithm](/figures/FF_algorithm.png)

The Forward-Forward Multi-Layer Perceptron as presented by [Hinton, 2022](https://arxiv.org/abs/2212.13345) (left), the Forward-Forward Convolutional Neural Network (right). The networks are optimised layer-wise. The positive and negative samples are fed into the first layer and via layer normalisation, we obtain orientation and length of the activation. The orientation is forwarded to the next layer as its input. The length is used for the computation of the goodness. Each layer is optimised so that positive samples have high goodness (> threshold) and negative samples have low goodness (< threshold). For inference, the sum of the goodness of all layers, excluding the first layer, needs to be determined for every possible label.

## Resource-efficient Framework
![Framework](/figures/FF_framework.png)

Proposed Self-adapting **SaFF-Net** framework. Key features of the training data and the hardware components are used for self-configuration. 
The fixed parameters for the pipeline are given by default or via an experiment file. After self-configuration, the **SaFF-Net** selects the best network configuration and starts training. Inference, postprocessing, calibration and pruning can be enabled.

### Efficiency
![Efficiency](/figures/FF_efficiency.png)

Classification on MNIST. ACC - Accuracy, AUC - Area Under the Receiver Operating characteristic, mAP - Mean Average Precision vs. Number of Parameters Comparison for MLP and FFA (top) and CNN and CFFA (bottom) with maximum batch size. Ours in orange.

## Citing
[1] Müller, J. P., & Kainz, B. (2024, October). Resource-efficient medical image analysis with self-adapting forward-forward networks. In International Workshop on Machine Learning in Medical Imaging (pp. 180-190). Cham: Springer Nature Switzerland.

@inproceedings{muller2024resource,
  title={Resource-efficient medical image analysis with self-adapting forward-forward networks},
  author={M{\"u}ller, Johanna P and Kainz, Bernhard},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={180--190},
  year={2024},
  organization={Springer}
}

```
#### Acknowledgements
(Some) HPC resources were provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR projects b143dc and b180dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.
```

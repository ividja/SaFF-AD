# SaFF-Net

We introduce a fast Self-adapting Forward-Forward Network (**SaFF-Net**) for medical imaging analysis, mitigating power consumption and resource limitations, which currently primarily stem from the prevalent reliance on back-propagation for model training and fine-tuning. Building upon the recently proposed Forward-Forward Algorithm (FFA), we introduce the Convolutional Forward-Forward Algorithm (CFFA), a parameter-efficient reformulation that is suitable for advanced image analysis and overcomes the speed and generalization constraints of the original FFA. To address hyper-parameter sensitivity we are also introducing a self-adapting framework, which tunes **SaFF-Net** to the optimal settings.
Our approach enables more effective model training and eliminates the previously essential requirement for an arbitrarily chosen Goodness function in FFA.
We evaluate our approach on several benchmarking datasets in comparison with standard Back-Propagation (BP) neural networks showing that FFA-based networks can compete with standard models with notably fewer parameters and function evaluations, especially, in one-shot scenarios and large batch sizes. 

[Paper]


from torchvision.transforms import Normalize
from torch.optim import Adam
from torch import nn
from source.models.forward_foward import CFFA_2D

data_keys = {
    'dataset': 'MNIST',
    'normalisation': 'auto',
    'modality': None,
    'dimensions': (1,28, 28), # (C, H, W)
}

resource_keys = {
    'memory': 80, # in GB
    'max_channels': 2024, # per layer
    'max_layers': 4,
}

train_mode = {
    'supervision': 'unsupervised',
    'data_augmentation': False,
    'batch_size': 'single', # single-batch training
    'mode': '2D', # only necessary for 3D volumes
    'warmup': False,
    'pruning': 'auto',
    'retrain_after_pruning': True,
    'early_stopping': ((5,1e-18), (10000,1e-18), (None,None)), # (Epochs,Layer-Iterations,Layers) with (patience, mind_delta)
    'max_epochs': 10,
    'max_layer_iterations': 1000,
    'peer_norm': False,
    'batch_norm': False,
}

fixed_parameters = {
    'optimiser': Adam,
    'learning_rate': 0.003,
    'application': 'classification',
    'loss_function': 'log_loss',
    'kernel_size': 3,
    'stride': [2, 1, 1, 1],
    'padding': 0,
    'activation': nn.ReLU(),
    'threshold': 4.0,
    'threshold_trainable': True,
    'model': CFFA_2D,
    'hidden_dims': [32,32,128,128],
    'log_dir_name': 'MNIST/' 
    }
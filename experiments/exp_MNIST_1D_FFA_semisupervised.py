from torchvision.transforms import Normalize
from torch.optim import Adam
from source.models.forward_foward import FFA


data_keys = {
    'dataset': 'MNIST',
    'normalisation': 'auto',
    'modality': None,
    'dimensions': (1,28, 28), # (C, H, W)
}

resource_keys = {
    'memory': 80, # in GB
    'max_neurons': 1000, # per layer
    'max_layers': 3,
}

train_mode = {
    'supervision': 'semi',
    'data_augmentation': False,
    'batch_size': 'single', # single-batch training
    'mode': '1D', 
    'warmup': True,
    'pruning': 'auto',
    'retrain_after_pruning': True,
    'early_stopping': ((5,1e-18),(10,1e-18),(2,1e-18)), # (Epochs,Layer-Iterations,Layers) with (patience, mind_delta)
    'max_epochs': 1,
    'max_layer_iterations': 1000,
    'peer_norm': True,
}

fixed_parameters = {
    'optimiser': Adam,
    'learning_rate': 0.03,
    'application': 'classification',
    'loss_function': 'log_loss',
    'threshold': 0.0,
    'threshold_trainable': True,
    'model': FFA,
    'log_dir_name': 'MNIST/' 
    }
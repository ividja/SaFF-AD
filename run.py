import logging
import os
import shutil
import argparse
import torch
import re
from importlib.machinery import SourceFileLoader
import numpy as np

from source.configuration import ModelConfiguration
from source.training import Trainer
from source.models.forward_foward import FFA, CFFA_2D, CFFA_3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# if necessary set the GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_dataloader(dataset):
    if dataset == 'MNIST':
        from source.data_loaders.MNIST_loader import MNIST_loader as loader
    elif dataset == 'MedMNIST':
        if exp_config.data_keys['dimensions'][0] > 3 :
            from source.data_loaders.MedMNIST_loader import MedMNIST_3D_loader as loader
        else:
            from source.data_loaders.MedMNIST_loader import MedMNIST_loader as loader
    elif dataset == 'VinDr-CXR':
        from source.data_loaders.VinDr_CXR_loader import VinDr_CXR_loader as loader
    else:
        raise ValueError('Unknown data %s' % dataset)
    return loader

def update_experiment_file(exp_config, trainer):
    exp_dir = os.path.join(exp_config.fixed_parameters['log_dir'], exp_config.__name__ +'.py')
    with open(exp_dir, 'r') as file:
        content = file.read()
    if exp_config.fixed_parameters['model'] in [FFA, CFFA_2D, CFFA_3D]:
        if exp_config.train_mode['mode'] == '1D':
            updated_content = re.sub(
                r"('max_neurons':\s*)(\d+)",
                r"\g<1>{}".format(trainer.net.layers[0].out_features),
                content,
                flags=re.MULTILINE
            )
        else:
            updated_content = re.sub(
                r"('max_channels':\s*)(\d+)",
                r"\g<1>{}".format(trainer.net.layers[0].out_channels),
                content,
                flags=re.MULTILINE
            )
        exp_dir_new = os.path.join(exp_config.fixed_parameters['log_dir'], exp_config.__name__ +'_optimised.py')
        with open(exp_dir_new, 'w') as new_file:
            new_file.write(updated_content)

def main(exp_config):
    logging.info('**************************************************************')
    logging.info('Running Experiment: %s', exp_config.__name__)
    logging.info('**************************************************************')
    
    logging.info('Loading dataloader...')
    loader = get_dataloader(exp_config.data_keys['dataset'])
    
    logging.info('Initialising model...')
    # merge all parameters for better handling
    parameters = {**exp_config.resource_keys, **exp_config.train_mode, **exp_config.fixed_parameters, **exp_config.data_keys}
    if exp_config.fixed_parameters['model'] in [FFA, CFFA_2D, CFFA_3D]:
        config = ModelConfiguration(**parameters)
        net = config.get_model()
    else:
        net = exp_config.fixed_parameters['model']

    logging.info('Initialising trainer...')
    trainer = Trainer(net, loader, **parameters)
    logging.info('Starting training...')
    trainer.train()
    logging.info('Finished training!')

    logging.info('Saving optimised experiment file ...')
    update_experiment_file(exp_config, trainer)

    logging.info('Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    parser.add_argument("SEED", type=int, help="Seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.SEED)
    np.random.seed(args.SEED)

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, config_file).load_module()

    log_dir = os.path.join('./log/', exp_config.fixed_parameters['log_dir_name'], exp_config.__name__)
    exp_config.fixed_parameters['log_dir'] = log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logging.info('Created experiment folder: %s', log_dir)
    else:
        logging.info('Experiment already exists!')
        ans = input('Do you want to overwrite it? [y/n]')
        if ans == 'n':
            exit(0)
        elif ans == 'y':
            logging.info('Using existing experiment folder: %s', log_dir)
            

    shutil.copy(exp_config.__file__, log_dir)
    logging.info('Copied configuration file to experiment folder!')

    main(exp_config)
import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from scipy.special import softmax
import os
from functools import reduce

from source.utils import EarlyStopper, augment_positive_samples
from source.supervision import generate_contrastive_pairs_x, generate_contrastive_pairs_y, generate_hybrid_contrastive_pairs
from source.models.forward_foward import ForwardForward, FF_ResNet
from source.models.cnn import CNN, ResNet18
from source.models.mlp import MLP

class Trainer():
    
    def __init__(self, net, loader, **kwargs):

        self.net = net
        self.loader = loader
        self.kwargs = kwargs
        
        # Direct assignments 
        self.log_dir_name = kwargs.get('log_dir')
        self.augment_transform = kwargs.get('data_augmentation')
        self.normalisation = kwargs.get('normalisation', 'auto')
        self.application = kwargs.get('application')
        self.mode = kwargs.get('mode')
        self.dimensions = kwargs.get('dimensions', [])
        self.hidden_dims = kwargs.get('hidden_dims', [32,32,128,128])
        self.optimiser = kwargs.get('optimiser')
        self.supervision = kwargs.get('supervision', 'full')
        self.max_memory_available = kwargs.get('memory')
        self.subset = kwargs.get('subset')
        self.batch_size = 100000 if kwargs.get('batch_size') == 'single' else kwargs.get('batch_size')
        self.resize = kwargs.get('resize')
        self.warmup = kwargs.get('warmup', True)
        self.pruning = kwargs.get('pruning', True)
        self.retrain_after_pruning = kwargs.get('retrain_after_pruning', True)
        self.loss_function = kwargs.get('loss_function')
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam)
        self.learning_rate = kwargs.get('learning_rate', 0.03)
        self.max_neurons = kwargs.get('max_neurons', 1000)
        self.max_channels = kwargs.get('max_channels', 2024)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.early_stopping = kwargs.get('early_stopping', ((5,1e-18),(10,1e-18),(2,1e-18)))

        # Conditional assignments
        self.labelling = self.application == 'classification'

        if self.loss_function == 'cross_entropy_loss':
            self.loss_function == nn.CrossEntropyLoss()

        self.flatten_input = self.mode == '1D' and self.dimensions and self.dimensions[0] <= 3
        self.slice_wise = self.dimensions and self.dimensions[0] > 3 and self.mode in ['2D', '1D']

        if self.supervision in ['full', 'semi']:
            self.pair_generator_function = generate_contrastive_pairs_x
        elif self.supervision == 'unsupervised':
            self.pair_generator_function = generate_hybrid_contrastive_pairs
            self.labelling = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_forward_forward = isinstance(self.net, ForwardForward)
        if self.is_forward_forward:
            self.net = self.net.to()

        if self.resize and self.resize[0] == 'resize':
            logging.info('Please include resizing in experiment file under data_keys[\'augment_transform\']!')
            exit()

        self.early_stopper_epochs = EarlyStopper(*self.early_stopping[0]) if self.early_stopping and self.early_stopping[0][0] is not None else None


    def calculate_memory_usage(self, train_loader, val_loader, test_loader):
        # Assuming 4 bytes per float (32-bit)
        float_size = 4

        if self.is_forward_forward:
            model_parameters = []
            for layer in range(len(self.net.layers)):
                model_parameters = filter(lambda p: p.requires_grad, self.net.layers[layer].parameters())
        else:
            model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        model_memory_usage = sum(p.numel() * float_size for p in model_parameters)

        input_sample, _ = next(iter(train_loader))
        input_memory_usage = input_sample.element_size() * input_sample.nelement()

        input_sample, _ = next(iter(test_loader))
        input_memory_usage += input_sample.element_size() * input_sample.nelement()

        input_sample, _ = next(iter(val_loader))
        input_memory_usage += input_sample.element_size() * input_sample.nelement()

        return model_memory_usage + input_memory_usage

    def determine_input_size(self, train_loader, val_loader, test_loader):
        memory_usage = self.calculate_memory_usage(train_loader, val_loader, test_loader)

        if memory_usage > self.max_memory_available * 10**9: # calculate_memory_usage returns bytes
            # If total memory usage exceeds GPU memory limit, switch to patches
            logging.info("Input size too large!")
            return "resize"
        else:
            # Otherwise, use full-size input
            logging.info("Using full-size input!")
            return "full_size"
        
    def interleave_slices(self, x, y, x_val, y_val):
        x = x.view(-1, 1, self.dimensions[1], self.dimensions[2])
        x_val = x_val.view(-1, 1, self.dimensions[1], self.dimensions[2])
        y = y.repeat_interleave(self.dimensions[0])
        y_val = y_val.repeat_interleave(self.dimensions[0])
        return x, y, x_val, y_val

    def warm_up(self, train_loader):
        logging.info('Warming up ...')
        x, y = next(iter(train_loader))
        x, y = x.cuda(), y.cuda()

        if self.mode == '1D':
            neurons_per_layer = [50,100,500,1000,2000,4000]
            neurons_per_layer = [num for num in neurons_per_layer if num <= self.max_neurons]
            thresholds = []
            for n in neurons_per_layer:
                logging.info('Precheck for %s Neurons per Layer' % n)
                self.net = self.net.__class__(self.net.dims[0], [n,n], **self.kwargs)
                x_pos, x_neg = self.pair_generator_function(x, y, train_loader.num_classes, self.dimensions,self.labelling)
                x_pos_test, x_neg_test = self.pair_generator_function(x, y, train_loader.num_classes, self.dimensions, self.labelling)
                if self.loss_function is None or self.loss_function == 'log_loss':
                    loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test)
                elif self.loss_function == 'contrastive':
                    x_pos_aug, x_pos_aug_test = augment_positive_samples(x_pos, x_pos_test, self.dimensions)
                    loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug, x_pos_aug_test)
                thresholds.append(self.net.layers[0].threshold)
            n_neurons = neurons_per_layer[thresholds.index(max(thresholds))]
            logging.info('Optimised Number of Neurons per Layer: %s' % n_neurons)
            self.net = self.net.__class__(self.net.dims[0], [n_neurons]*self.net.max_layers, **self.kwargs)
        
        elif self.mode == '2D':
            first_layer_channels = [16,32,64]
            thresholds = []
            for n in first_layer_channels:
                channels = [n*2**layer for layer in range(self.net.max_layers) if n*2**layer <= self.max_channels]
                logging.info('Precheck for %s Channel dimensions in first Layer' % n)
                self.net = self.net.__class__(self.net.channels[0], channels[:2], **self.kwargs)
                x_pos, x_neg = self.pair_generator_function(x, y, train_loader.num_classes, self.labelling)
                x_pos_test, x_neg_test = self.pair_generator_function(x, y, train_loader.num_classes, self.labelling)
                loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test)
                thresholds.append(self.net.layers[0].threshold)
            first_channels = first_layer_channels[thresholds.index(max(thresholds))]
            logging.info('Optimised number of first Channel dimensions per Layer: %s' % first_channels)
            channels = [n*2**layer for layer in range(self.net.max_layers) if n*2**layer <= self.max_channels]
            self.net = self.net.__class__(self.net.channels[0], channels, **self.kwargs)
            

    def prune(self):
        logging.info('Pruning ...')
        if self.is_forward_forward:
            new_layers = self.net.layers.copy()
            for layer in new_layers:
                for name, module in layer.named_modules():
                # prune 20% of connections in all 2D-conv layers
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=0.05)
                        prune.remove(module, 'weight')
                    # prune 40% of connections in all linear layers
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=0.4)
                        prune.remove(module, 'weight')

            self.net.layers = new_layers

    def fine_tune(self, train_loader, val_loader):
        logging.info('Fine-tuning ...')

        x, y = next(iter(train_loader))
        x, y = x.cuda(), y.cuda()

        x_val, y_val = next(iter(val_loader))
        x_val, y_val = x.cuda(), y.cuda()

        if self.is_forward_forward:
            if self.slice_wise == True:
                x, y, x_val, y_val = self.interleave_slices(x, y, x_val, y_val)
                if self.mode == '1D':
                    x= x.flatten(start_dim=1)
                    x_val= x_val.flatten(start_dim=1)
            else:
                if self.mode == '3D':
                    # con3d needs 5D input
                    x = x.view(-1, 1, self.dimensions[0], self.dimensions[1], self.dimensions[2])
                    x_val = x_val.view(-1, 1, self.dimensions[0], self.dimensions[1], self.dimensions[2])
                    
            x_pos, x_neg = self.pair_generator_function(x, y, train_loader.num_classes, self.dimensions, self.labelling)
            x_pos_test, x_neg_test = self.pair_generator_function(x_val, y_val, val_loader.num_classes, self.dimensions, self.labelling)

        if self.is_forward_forward:
            for i in range(len(self.net.layers)):
                self.net.layers[i].opt = self.optimiser(self.net.layers[i].parameters(), lr=self.learning_rate*1e-3)
                self.net.layers[i].num_iter = 100
                self.net.layers[i].early_stopper_iterations = EarlyStopper(10,1e-18)
        
            if self.loss_function is None or self.loss_function == 'log_loss':
                    loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test)
            elif self.loss_function == 'contrastive':
                x_pos_aug, x_pos_aug_test = augment_positive_samples(x_pos, x_pos_test, self.dimensions)
                loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug, x_pos_aug_test)

        
    def count_parameters(self):
        if self.is_forward_forward:
            params = 0
            for layer in range(len(self.net.layers)):
                model_parameters = filter(lambda p: p.requires_grad, self.net.layers[layer].parameters())
                params += sum([np.prod(p.size()) for p in model_parameters])
        else:
            model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
        
        logging.info('# Parameters: %s' % params)
        return params


    def evaluate(self, test_loader):
        num_classes = test_loader.num_classes
        metrics = {'acc': [], 'auroc': [], 'prec': [], 'auroc_softmax': []}

        for x, y in test_loader:
            x_te, y_te = x.cuda(), y.cuda()
            y_te = torch.squeeze(y_te)
            if y_te.dim() == 2:
                y_te = torch.argmax(y_te, dim=1)

            if self.slice_wise:
                x_te = x_te.view(-1, 1, *self.dimensions[1:])
                y_te = y_te.repeat_interleave(self.dimensions[0])
                if self.mode == '1D':
                    x_te = x_te.flatten(start_dim=1)
            elif self.mode == '3D':
                x_te = x_te.view(-1, 1, *self.dimensions)

            if self.is_forward_forward or isinstance(self.net, FF_ResNet):
                y_pred, goodness = self.net.predict(x_te, num_classes)
                goodness_softmax = torch.nn.functional.softmax(goodness, dim=1).detach().cpu().numpy() 
            else:
                y_pred, goodness = self.net.predict(x_te)
                goodness_softmax = np.zeros((y_te.size(0), num_classes))

            y_te_np = y_te.detach().cpu().numpy() 
            y_pred_np = y_pred.detach().cpu().numpy() 
            y_te_oh = torch.nn.functional.one_hot(y_te, num_classes=num_classes).detach().cpu().numpy() 


            metrics['acc'].append(accuracy_score(y_te_np, y_pred_np))
            metrics['auroc'].append(roc_auc_score(y_te_oh, goodness.detach().cpu().numpy() , multi_class="ovr", average="micro"))
            metrics['prec'].append(average_precision_score(y_te_oh, goodness_softmax, average="micro"))
            if self.is_forward_forward or isinstance(self.net, FF_ResNet):
                metrics['auroc_softmax'].append(roc_auc_score(y_te_oh, goodness_softmax, multi_class="ovr", average="micro"))
            else:
                metrics['auroc_softmax'].append(0.0)  

        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f'test {metric_name}: {mean_val:.4f}    {std_val:.4f}')

        return metrics['acc'][0], metrics['auroc'][0], metrics['prec'][0], metrics['auroc_softmax'][0]


    def train(self):

        # application
        train_loader = self.loader(self.batch_size, subset=self.subset, split='train', flatten=self.flatten_input, 
                                          norm_transform=self.normalisation, augment_transform=self.augment_transform, num_workers=8)
        val_loader = self.loader(100, subset=self.subset, split='val', flatten=self.flatten_input, 
                                          norm_transform=self.normalisation, augment_transform=self.augment_transform, num_workers=8)
        test_loader = self.loader(100, subset=self.subset, split='test', flatten=self.flatten_input, 
                                          norm_transform=self.normalisation, augment_transform=self.augment_transform, num_workers=8)

        if not self.is_forward_forward:
            if self.net == ResNet18:
                self.net = self.net(train_loader.num_classes, self.loss_function, self.optimiser, self.learning_rate, **self.kwargs)
                self.net = self.net.to(self.device)
            elif self.net == CNN:
                if self.slice_wise == True:
                    self.net = self.net(1, train_loader.num_classes, self.hidden_dims, self.loss_function, self.optimiser, self.learning_rate, **self.kwargs)
                else:
                    self.net = self.net(self.dimensions[0], train_loader.num_classes, self.hidden_dims, self.loss_function, self.optimiser, self.learning_rate, **self.kwargs)
                self.net = self.net.to(self.device)
            elif self.net == MLP:
                self.net = self.net(reduce(lambda x, y: x*y, self.dimensions), train_loader.num_classes, self.hidden_dims, self.loss_function, self.optimiser, self.learning_rate)
                self.net = self.net.to(self.device)
            elif self.net == FF_ResNet:
                # FF_ResNet architecture is hardcoded here
                if self.slice_wise == True:
                    self.net = self.net(2, self.loss_function, self.optimiser, self.learning_rate, **self.kwargs)
                elif self.supervision == 'unsupervised' or self.application == 'pretraining':
                    self.net = self.net(self.dimensions[0], self.loss_function, self.optimiser, self.learning_rate, **self.kwargs)
                else:
                    self.net = self.net(self.dimensions[0]+1, self.loss_function, self.optimiser, self.learning_rate, **self.kwargs)
                self.net = self.net.to(self.device)

        if self.is_forward_forward:
            input_size = self.determine_input_size(train_loader, val_loader, test_loader)
            if input_size == "resize":
                # add resize/patching method
                pass
            if self.warmup:
                    self.warm_up(train_loader)

        parameters = self.count_parameters()

        for i in range(self.max_epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()

                x_val, y_val = next(iter(val_loader))
                x_val, y_val = x_val.cuda(), y_val.cuda()

                if self.is_forward_forward or isinstance(self.net, FF_ResNet):
                    if self.slice_wise == True:
                        x, y, x_val, y_val = self.interleave_slices(x, y, x_val, y_val)
                        if self.mode == '1D':
                            x= x.flatten(start_dim=1)
                            x_val= x_val.flatten(start_dim=1)
                    else:
                        if self.mode == '3D':
                            # con3d needs 5D input
                            x = x.view(-1, 1, self.dimensions[0], self.dimensions[1], self.dimensions[2])
                            x_val = x_val.view(-1, 1, self.dimensions[0], self.dimensions[1], self.dimensions[2])
                    
                    if self.supervision == 'semi':
                        # assume 10 % unlabelled
                        count = int(x.shape[0] * 0.1)
                        tensor = torch.zeros(x.shape[0], dtype=torch.bool)
                        true_indices = torch.randperm(x.shape[0])[:count]
                        tensor[true_indices] = True

                        x_labelled = x[tensor]
                        y_labelled = y[tensor]
                        x_unlabelled = x[~tensor]
                        y_unlabelled = y[~tensor]
                        x_pos_labelled, x_neg_labelled = self.pair_generator_function(x_labelled, y_labelled, train_loader.num_classes, self.dimensions, True)
                        if self.mode == '1D':
                            x_pos_unlabelled, x_neg_unlabelled = self.pair_generator_function(x_unlabelled, y_unlabelled, train_loader.num_classes,self.dimensions, False, True)
                        elif self.mode == '2D':
                            x_pos_unlabelled, x_neg_unlabelled = self.pair_generator_function(x_unlabelled, y_unlabelled, train_loader.num_classes, self.dimensions,True, True)
                        
                        x_pos, x_neg = torch.concat([x_pos_labelled, x_pos_unlabelled], 0), torch.concat([x_neg_labelled, x_neg_unlabelled], 0)
                    else:
                        x_pos, x_neg = self.pair_generator_function(x, y, train_loader.num_classes, self.dimensions,self.labelling)
                    
                    x_pos_test, x_neg_test = self.pair_generator_function(x_val, y_val, val_loader.num_classes, self.dimensions,self.labelling)
                    
                    if self.loss_function is None or self.loss_function == 'log_loss':
                        loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test)
                    elif self.loss_function == 'contrastive':
                        x_pos_aug, x_pos_aug_test = augment_positive_samples(x_pos, x_pos_test, self.dimensions)
                        loss_test = self.net.train(x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug, x_pos_aug_test)
                else:
                    if self.slice_wise == True:
                        x, y, x_val, y_val = self.interleave_slices(x, y, x_val, y_val)
                    y = y.squeeze()
                    y_val = y_val.squeeze()
                    if y.dim() == 2:
                        y = torch.argmax(y, dim=1)
                    if y_val.dim() == 2:
                        y_val = torch.argmax(y_val, dim=1)
                    loss_test = self.net.train(x, y, x_val, y_val)
            
            logging.info('Epoch %s: Test Loss: %s' % (i, loss_test))

            if hasattr(self, 'early_stopper_epochs'):
                if self.early_stopper_epochs.early_stop(loss_test):    
                    logging.info('Stopped early at epoch %s ...' % i)         
                    break
                if self.early_stopper_epochs.counter == 0:
                    self.net.save_checkpoint(self.log_dir_name, 'model_best_loss')
            
            if np.isnan(loss_test):
                reason = f"NaN detected - Test Loss is NaN at epoch {i}!"
                logging.error(reason)
                raise ValueError(reason)
                
        if self.is_forward_forward:
            logging.info('Reduced Parameters')
            self.count_parameters()

        if self.application == 'classification' and (self.supervision == 'full' or self.supervision == 'semi'):
            acc, auroc, prec, auroc_softmax = self.evaluate(test_loader)
            self.net.save_checkpoint(self.log_dir_name, 'model_trained')

            if self.pruning == 'auto' or self.pruning == True:
                self.prune()
                acc, auroc, prec, auroc_softmax = self.evaluate(test_loader)
                self.net.save_checkpoint(self.log_dir_name, 'model_pruned')

                if self.retrain_after_pruning:
                    self.fine_tune(train_loader, val_loader)
                    acc, auroc, prec, auroc_softmax = self.evaluate(test_loader)

                    self.net.save_checkpoint(self.log_dir_name, 'model_retrained')
            
            return acc, auroc, prec, auroc_softmax, parameters
        
        elif self.application == 'pretraining' or self.supervision == 'unsupervised':
            self.net.save_checkpoint(self.log_dir_name, 'model_pretrained')
        


        

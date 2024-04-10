import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from source.utils import EarlyStopper
from source.utils import log_loss


class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias)
        self.activation = kwargs.get('activation', nn.LeakyReLU())
        if self.activation is False:
            self.activation = nn.Identity()
        threshold_init = kwargs.get('threshold', 2.0)
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.threshold.requires_grad = kwargs.get('threshold_trainable', False)
        self.threshold_restricted = False
        
        self.num_iter = kwargs.get('max_layer_iterations', 1000)
        self.opt = kwargs.get('optimiser', torch.optim.Adam)(self.parameters(), lr=kwargs.get('learning_rate', 0.03))
        self.loss_function = kwargs.get('loss_function', None)

        # Early Stopping
        self.early_stopping = kwargs.get('early_stopping',None)
        if self.early_stopping is not None and self.early_stopping[1][0] is not None:
            # Layer-Iterations
            self.early_stopper_iterations = EarlyStopper(self.early_stopping[1][0], self.early_stopping[1][1])
        
        self.peer_norm_enabled = kwargs.get('peer_norm', True)
        if self.peer_norm_enabled:
            self.peer_norm_weight = 0.03
            self.momentum = 0.9
            self.running_mean = torch.zeros(out_features).cuda()  

        self.loss_function = kwargs.get('loss_function', None)
        if self.loss_function is None or self.loss_function == 'log_loss':
            self.loss = log_loss
        elif self.loss_function == 'contrastive':
            self.loss = nn.TripletMarginLoss()
    
    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
    
    def peer_norm_loss(self, x):
        mean_activation = x.mean(0)
        self.running_mean = self.running_mean * self.momentum + mean_activation * (1 - self.momentum)
        return torch.mean((mean_activation - self.running_mean) ** 2) + 1e-6 

    def train(self, x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug=None, x_pos_aug_test=None):

        for i in tqdm(range(self.num_iter)):

            if self.loss_function == 'log_loss':
                x = torch.cat((x_pos, x_neg),0)
                x_val = torch.cat((x_pos_test, x_neg_test),0)
                # Forward pass
                forward_x = self.forward(x)
                with torch.no_grad():
                    forward_x_val = self.forward(x_val)
                forward_pos, forward_neg = torch.split(forward_x, [x_pos.size(0), x_neg.size(0)], 0)
                forward_pos_test, forward_neg_test = torch.split(forward_x_val, [x_pos_test.size(0), x_neg_test.size(0)], 0)
                g_pos = forward_pos.pow(2).mean(1)
                g_neg = forward_neg.pow(2).mean(1)
                g_pos_test = forward_pos_test.pow(2).mean(1)
                g_neg_test = forward_neg_test.pow(2).mean(1)
                loss = self.loss(g_pos, g_neg, self.threshold)
            elif self.loss_function == 'contrastive':
                x = torch.cat((x_pos, x_neg, x_pos_aug),0)
                x_val = torch.cat((x_pos_test, x_neg_test, x_pos_aug_test),0)
                # Forward pass
                forward_x = self.forward(x)
                with torch.no_grad():
                    forward_x_val = self.forward(x_val)
                forward_pos, forward_neg, forward_pos_aug= torch.split(forward_x, [x_pos.size(0), x_neg.size(0), x_pos_aug.size(0)], 0)
                forward_pos_test, forward_neg_test, forward_pos_aug_test = torch.split(forward_x_val, [x_pos_test.size(0), x_neg_test.size(0), x_pos_aug_test.size(0)], 0)
                loss = self.loss(forward_pos, forward_pos_aug, forward_neg)

            if self.peer_norm_enabled:
                loss += self.peer_norm_weight * self.peer_norm_loss(forward_pos)

            with torch.no_grad():
                if self.loss_function == 'log_loss':
                    loss_test = self.loss(g_pos_test, g_neg_test, self.threshold)
                elif self.loss_function == 'contrastive':
                    loss_test = self.loss(forward_pos_test, forward_pos_aug_test, forward_neg_test)
                if self.peer_norm_enabled:
                    loss_test += self.peer_norm_weight * self.peer_norm_loss(forward_pos_test)
            
            if hasattr(self, 'early_stopper_iterations'):          
                if self.early_stopper_iterations.early_stop(loss_test):    
                    logging.info('Stopped early at iteration %s ...' % i)         
                    break

            self.opt.zero_grad()
            # calc derivatives for local updates, could be replaced by other estimation and optimisation methods
            loss.backward(retain_graph=True)
            self.opt.step()
        
        if x_pos_aug is not None:
            return self.forward(x_pos).detach(), self.forward(x_neg).detach(), self.forward(x_pos_test).detach(), self.forward(x_neg_test).detach(), self.forward(x_pos_aug).detach(), self.forward(x_pos_aug_test).detach(), loss_test.item()
        else:
            return self.forward(x_pos).detach(), self.forward(x_neg).detach(), self.forward(x_pos_test).detach(), self.forward(x_neg_test).detach(), None, None, loss_test.item()

class FF_ConvLayer_2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernelsize: int, paddings: int, strides: int, bias=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=kernelsize, stride=strides, 
                         padding=paddings, bias=bias)
        self.activation = kwargs.get('activation', nn.ReLU())
        if self.activation is False:
            self.activation = nn.Identity()
        threshold_init = kwargs.get('threshold', 2.0)
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.threshold.requires_grad = kwargs.get('threshold_trainable', False)
        self.threshold_restricted = False
        
        self.num_iter = kwargs.get('max_layer_iterations', 1000)
        self.opt = kwargs.get('optimiser', torch.optim.Adam)(self.parameters(), lr=kwargs.get('learning_rate', 0.03))

        # Early Stopping
        self.early_stopping = kwargs.get('early_stopping',None)
        if self.early_stopping is not None and self.early_stopping[1][0] is not None:
            # Layer-Iterations
            self.early_stopper_iterations = EarlyStopper(self.early_stopping[1][0], self.early_stopping[1][1])
        
        self.batch_norm_enabled = kwargs.get('batch_norm', True)
        if self.batch_norm_enabled:
            self.batch_norm_weight = 3e-5

        self.loss_function = kwargs.get('loss_function', None)
        if self.loss_function is None or self.loss_function == 'log_loss':
            self.loss = log_loss
        elif self.loss_function == 'contrastive':
            self.loss = nn.TripletMarginLoss()

        #self.dropout = nn.Dropout2d(0.05)
    
    def forward(self, x):
        #x = self.dropout(x)
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = self._conv_forward(x_direction, self.weight, self.bias)
        return self.activation(x_direction)
    
    def batch_norm_loss(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        mean_loss = torch.mean(batch_mean ** 2)  # Encourage mean to be 0
        var_loss = torch.mean((batch_var - 1) ** 2)  # Encourage variance to be 1
        bn_loss = mean_loss + var_loss
        return bn_loss + 1e-6 

    def train(self, x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug=None, x_pos_aug_test=None):

        for i in tqdm(range(self.num_iter)):

            if self.loss_function == 'log_loss':
                x = torch.cat((x_pos, x_neg),0)
                x_val = torch.cat((x_pos_test, x_neg_test),0)
                # Forward pass
                forward_x = self.forward(x)
                with torch.no_grad():
                    forward_x_val = self.forward(x_val)
                forward_pos, forward_neg = torch.split(forward_x, [x_pos.size(0), x_neg.size(0)], 0)
                forward_pos_test, forward_neg_test = torch.split(forward_x_val, [x_pos_test.size(0), x_neg_test.size(0)], 0)
                g_pos = forward_pos.pow(2).mean(1)
                g_neg = forward_neg.pow(2).mean(1)
                g_pos_test = forward_pos_test.pow(2).mean(1)
                g_neg_test = forward_neg_test.pow(2).mean(1)
                loss = self.loss(g_pos, g_neg, self.threshold)
            elif self.loss_function == 'contrastive':
                x = torch.cat((x_pos, x_neg, x_pos_aug),0)
                x_val = torch.cat((x_pos_test, x_neg_test, x_pos_aug_test),0)
                # Forward pass
                forward_x = self.forward(x)
                with torch.no_grad():
                    forward_x_val = self.forward(x_val)
                forward_pos, forward_neg, forward_pos_aug= torch.split(forward_x, [x_pos.size(0), x_neg.size(0), x_pos_aug.size(0)], 0)
                forward_pos_test, forward_neg_test, forward_pos_aug_test = torch.split(forward_x_val, [x_pos_test.size(0), x_neg_test.size(0), x_pos_aug_test.size(0)], 0)
                loss = self.loss(forward_pos, forward_pos_aug, forward_neg)

            if self.batch_norm_enabled:
                loss += self.batch_norm_weight * self.batch_norm_loss(forward_pos)

            with torch.no_grad():
                if self.loss_function == 'log_loss':
                    loss_test = self.loss(g_pos_test, g_neg_test, self.threshold)
                elif self.loss_function == 'contrastive':
                    loss_test = self.loss(forward_pos_test, forward_pos_aug_test, forward_neg_test)
                if self.batch_norm_enabled:
                    loss_test += self.batch_norm_weight * self.batch_norm_loss(forward_pos_test)
            
            if hasattr(self, 'early_stopper_iterations'):            
                #logging.info('Loss: %s ' % loss_test)
                if self.early_stopper_iterations.early_stop(loss_test):     
                    logging.info('Stopped early at iteration %s ...' % i)         
                    break

            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
        
        if x_pos_aug is not None:
            return self.forward(x_pos).detach(), self.forward(x_neg).detach(), self.forward(x_pos_test).detach(), self.forward(x_neg_test).detach(), self.forward(x_pos_aug).detach(), self.forward(x_pos_aug_test).detach(), loss_test.item()
        else:
            return self.forward(x_pos).detach(), self.forward(x_neg).detach(), self.forward(x_pos_test).detach(), self.forward(x_neg_test).detach(), None, None, loss_test.item()
    

class FF_ConvLayer_3D(nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernelsize: int, paddings: int, strides: int, bias=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=kernelsize, stride=strides, 
                         padding=paddings, bias=bias)
        self.activation = kwargs.get('activation', nn.ReLU())
        threshold_init = kwargs.get('threshold', 2.0)
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.threshold.requires_grad = kwargs.get('threshold_trainable', False)
        self.threshold_restricted = False
        
        self.num_iter = kwargs.get('max_layer_iterations', 1000)
        self.opt = kwargs.get('optimiser', torch.optim.Adam)(self.parameters(), lr=kwargs.get('learning_rate', 0.03))

        # Early Stopping
        self.early_stopping = kwargs.get('early_stopping',None)
        if self.early_stopping is not None and self.early_stopping[1][0] is not None:
            # Layer-Iterations
            self.early_stopper_iterations = EarlyStopper(self.early_stopping[1][0], self.early_stopping[1][1])
        
        self.batch_norm_enabled = kwargs.get('batch_norm', True)
        if self.batch_norm_enabled:
            self.batch_norm_weight = 3e-5

        self.loss_function = kwargs.get('loss_function', None)
        if self.loss_function is None or self.loss_function == 'log_loss':
            self.loss = log_loss
        else:
            pass
    
    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = self._conv_forward(x_direction, self.weight, self.bias)
        return self.activation(x_direction)
    
    def batch_norm_loss(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        mean_loss = torch.mean(batch_mean ** 2)  # Encourage mean to be 0
        var_loss = torch.mean((batch_var - 1) ** 2)  # Encourage variance to be 1
        bn_loss = mean_loss + var_loss
        return bn_loss + 1e-6 

    def train(self, x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug=None, x_pos_aug_test=None):

        for i in tqdm(range(self.num_iter)):

            if self.loss_function == 'log_loss':
                x = torch.cat((x_pos, x_neg),0)
                x_val = torch.cat((x_pos_test, x_neg_test),0)
                # Forward pass
                forward_x = self.forward(x)
                with torch.no_grad():
                    forward_x_val = self.forward(x_val)
                forward_pos, forward_neg = torch.split(forward_x, [x_pos.size(0), x_neg.size(0)], 0)
                forward_pos_test, forward_neg_test = torch.split(forward_x_val, [x_pos_test.size(0), x_neg_test.size(0)], 0)
                g_pos = forward_pos.pow(2).mean(1)
                g_neg = forward_neg.pow(2).mean(1)
                g_pos_test = forward_pos_test.pow(2).mean(1)
                g_neg_test = forward_neg_test.pow(2).mean(1)
                loss = self.loss(g_pos, g_neg, self.threshold)
            elif self.loss_function == 'contrastive':
                x = torch.cat((x_pos, x_neg, x_pos_aug),0)
                x_val = torch.cat((x_pos_test, x_neg_test, x_pos_aug_test),0)
                # Forward pass
                forward_x = self.forward(x)
                with torch.no_grad():
                    forward_x_val = self.forward(x_val)
                forward_pos, forward_neg, forward_pos_aug= torch.split(forward_x, [x_pos.size(0), x_neg.size(0), x_pos_aug.size(0)], 0)
                forward_pos_test, forward_neg_test, forward_pos_aug_test = torch.split(forward_x_val, [x_pos_test.size(0), x_neg_test.size(0), x_pos_aug_test.size(0)], 0)
                loss = self.loss(forward_pos, forward_pos_aug, forward_neg)

            if self.batch_norm_enabled:
                loss += self.batch_norm_weight * self.batch_norm_loss(forward_pos)

            with torch.no_grad():
                if self.loss_function == 'log_loss':
                    loss_test = self.loss(g_pos_test, g_neg_test, self.threshold)
                elif self.loss_function == 'contrastive':
                    loss_test = self.loss(forward_pos_test, forward_pos_aug_test, forward_neg_test)
                if self.batch_norm_enabled:
                    loss_test += self.batch_norm_weight * self.batch_norm_loss(forward_pos_test)
            
            if hasattr(self, 'early_stopper_iterations'):            
                #logging.info('Loss: %s ' % loss_test)
                if self.early_stopper_iterations.early_stop(loss_test):     
                    logging.info('Stopped early at iteration %s ...' % i)         
                    break

            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
        
        if x_pos_aug is not None:
            return self.forward(x_pos).detach(), self.forward(x_neg).detach(), self.forward(x_pos_test).detach(), self.forward(x_neg_test).detach(), self.forward(x_pos_aug).detach(), self.forward(x_pos_aug_test).detach(), loss_test.item()
        else:
            return self.forward(x_pos).detach(), self.forward(x_neg).detach(), self.forward(x_pos_test).detach(), self.forward(x_neg_test).detach(), None, None, loss_test.item()
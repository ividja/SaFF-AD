import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from source.utils import EarlyStopper
from source.supervision import overlay_y_on_x, concat_y_and_x
from source.models.layers import FF_Layer, FF_ConvLayer_2D, FF_ConvLayer_3D



class ForwardForward(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        
        # Early Stopping
        self.early_stopping = kwargs.get('early_stopping',None)
        if self.early_stopping is not None and self.early_stopping[2][0] is not None:
            # Layers
            self.early_stopper_layers = EarlyStopper(self.early_stopping[2][0], self.early_stopping[2][1])
        
        # Network Parameters
        self.peer_norm_enabled = kwargs.get('peer_norm', True)
        self.max_layers = kwargs.get('max_layers', 6)
        self.max_neurons = kwargs.get('max_neurons', 2000)
        self.max_layer_iterations = kwargs.get('max_layer_iterations', 1000)


        # Training Modes
        self.pruning = kwargs.get('pruning', None)
        self.warmup = kwargs.get('warmup', None)
        self.batch_size = kwargs.get('batch_size', None)
        if self.batch_size == 'single':
            # change if necessary
            self.batch_size = 100000
        
    def build(self):
        return

    def predict(self, x, num_classes):
        return 

    def train(self, x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug = None, x_neg_aug = None):                
        
        h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug = x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug, x_neg_aug

        for i, layer in enumerate(self.layers):
            logging.info('Training layer %s  ...' % i)
            if isinstance(layer, FF_Layer) or isinstance(layer, FF_ConvLayer_2D) or isinstance(layer, FF_ConvLayer_3D): 
                
                h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = layer.train(h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug)
                logging.info('Opt. Threshold: %s' % layer.threshold.item())
                logging.info('Loss: %s' % loss_test)
                # we need 2 layers ar least
                if i > 1:
                    if hasattr(self, 'early_stopper_layers'):
                        if self.early_stopper_layers.early_stop(loss_test):    
                            next =  i+1
                            if next < len(self.layers):
                                for j in range(0, len(self.layers) - next):
                                    del self.layers[-1]
                                    logging.info('Deleted layer: %s ' % (next+j))
                        
                            logging.info('Stopped early at layer %s  ...' % i)         
                            break
            else:
                h_pos, h_neg, h_pos_test, h_neg_test = layer(h_pos), layer(h_neg), layer(h_pos_test), layer(h_neg_test)
                if h_pos_aug is not None:
                    h_pos_aug, h_neg_aug = layer(h_pos_aug), layer(h_neg_aug)
        
        return loss_test

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_norm_vectors(self, x: torch.Tensor):
        norm_vectors = []
        for layer in self.layers:
            x = layer(x)
            norm_vectors.append(x)
        return norm_vectors
    
    def save_checkpoint(self, path, name):

        checkpoint = {
            'model_state_dict': [self.layers[i].state_dict() for i in range(len(self.layers)) if hasattr(self.layers[i], 'state_dict')],
            'optimizer_state_dict': [self.layers[i].opt.state_dict() for i in range(len(self.layers)) if hasattr(self.layers[i], 'opt')]
        }
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(checkpoint, path + '/' + name + '.pt')

    def load_checkpoint(self, path, name):

        checkpoint = torch.load(path + '/' + name + '.pt')
        
        # Load optimizer states for each layer
        for i, layer in enumerate(checkpoint['optimizer_state_dict']):
            if hasattr(self.layers[i], 'state_dict'):
                self.layers[i].load_state_dict(checkpoint['model_state_dict'][i] )
            if hasattr(self.layers[i], 'opt'):
                self.layers[i].opt.load_state_dict(checkpoint['optimizer_state_dict'][i])

        next =  i+1
        for j in range(0, len(checkpoint['optimizer_state_dict']) - next):
            del self.layers[-1]
            logging.info('Deleted empty layer: %s ' % (next+j))


class FFA(ForwardForward):
    
    def __init__(self, input_size: int, dims: list, **kwargs):
        super().__init__(**kwargs)

        # Network Setup
        self.dims = [input_size] 
        if dims is None:
            self.dims += [self.max_neurons] * self.max_layers
        else:
            self.dims += dims

        self.build()

    def build(self):
        self.layers = []
        for d in range(len(self.dims) - 1):
            self.layers += [FF_Layer(self.dims[d], self.dims[d + 1], bias=True, **self.kwargs).cuda()]


    def predict(self, x, num_classes):
        goodness_per_label = []
        for label in range(num_classes):
            h = overlay_y_on_x(x, torch.tensor(label).type(torch.IntTensor),num_classes)
            goodness = []
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i > 0:
                    goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1), goodness_per_label
    
    def embed(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        return h


class CFFA_2D(ForwardForward):
    
    def __init__(self, input_size: int, dims: list, **kwargs):
        super().__init__(**kwargs)

        # Network Setup
        self.channels = [input_size] 

        if dims is None:
            self.channels += [32,32,128,128]
        else:
            self.channels += dims

        if not isinstance(kwargs.get('kernel_size', 3), list):
            self.list_kernel_size = [kwargs.get('kernel_size', 3)] * (len(self.channels) + 1)
        else:   
            self.list_kernel_size = kwargs.get('kernel_size', None)
        
        if not isinstance(kwargs.get('stride', 2), list):
            self.list_stride = [kwargs.get('stride', 2)] * (len(self.channels) + 1)
        else:
            self.list_stride = kwargs.get('stride', None)
        
        if not isinstance(kwargs.get('padding', 0), list):
            self.list_padding = [kwargs.get('padding', 1)] * (len(self.channels) + 1)
        else:
            self.list_padding = kwargs.get('padding', None)

        if kwargs.get('application', None) == 'pretraining':
            self.pretraining = True
        else:
            self.pretraining = False
        
        self.build()

    def build(self):
        self.layers = []
        for d in range(len(self.channels) - 1):
            self.layers += [FF_ConvLayer_2D(in_channels=self.channels[d], out_channels=self.channels[d + 1], kernelsize=self.list_kernel_size[d], paddings=self.list_padding[d], strides=self.list_stride[d], bias=True, **self.kwargs).cuda()]

        #if self.pretraining:
            #self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            #self.layers.append(FF_Layer(self.channels[-1], 64, bias=True, **self.kwargs).cuda())
        

    def predict(self, x, num_classes):
        goodness_per_label = []
        for label in range(num_classes):
            label = torch.tensor([label] * x.size(0)).cuda()
            h = concat_y_and_x(x, label, num_classes)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum([torch.sum(goodness[i],(1,2)) for i in range(len(goodness))]).unsqueeze(0)]
        goodness_per_label = torch.concat(goodness_per_label,0)
        return goodness_per_label.argmax(0), goodness_per_label.T
    
    def embed(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, FF_Layer):
                pass
                #h = layer(h.flatten(start_dim=1))
            else:
                h = layer(h)
        return h
    
class CFFA_3D(ForwardForward):
    
    def __init__(self, input_size: int, dims: list, **kwargs):
        super().__init__(**kwargs)

        # Network Setup
        self.channels = [input_size] 

        if dims is None:
            self.channels += [32,32,128,128]
        else:
            self.channels += dims

        if not isinstance(kwargs.get('kernel_size', 3), list):
            self.list_kernel_size = [kwargs.get('kernel_size', 3)] * (len(self.channels) + 1)
        else:   
            self.list_kernel_size = kwargs.get('kernel_size', None)
        
        if not isinstance(kwargs.get('stride', 2), list):
            self.list_stride = [kwargs.get('stride', 2)] * (len(self.channels) + 1)
        else:
            self.list_stride = kwargs.get('stride', None)
        
        if not isinstance(kwargs.get('padding', 0), list):
            self.list_padding = [kwargs.get('padding', 1)] * (len(self.channels) + 1)
        else:
            self.list_padding = kwargs.get('padding', None)
        
        self.build()

    def build(self):
        self.layers = []
        for d in range(len(self.channels) - 1):
            self.layers += [FF_ConvLayer_3D(in_channels=self.channels[d], out_channels=self.channels[d + 1], kernelsize=self.list_kernel_size[d], paddings=self.list_padding[d], strides=self.list_stride[d], bias=True, **self.kwargs).cuda()]

    def predict(self, x, num_classes):
        goodness_per_label = []
        for label in range(num_classes):
            label = torch.tensor([label] * x.size(0)).cuda()
            h = concat_y_and_x(x, label, num_classes)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum([torch.sum(goodness[i],(1,2,3)) for i in range(len(goodness))]).unsqueeze(0)]
        goodness_per_label = torch.concat(goodness_per_label,0)
        return goodness_per_label.argmax(0), goodness_per_label.T


class FF_ResNet(nn.Module):
    def __init__(self, input_channels, loss, opt, lr, **kwargs):
        super(FF_ResNet, self).__init__()
        self.kwargs = kwargs
        #self.kwargs['activation'] = False

        self.layers=[]
        
        # Initial convolution
        self.conv1 = FF_ConvLayer_2D(input_channels, 64, kernelsize=3, strides=1, paddings=1, bias=True, **self.kwargs)
        self.layers.append(self.conv1)

        # Layer 1
        self.conv2 = FF_ConvLayer_2D(64, 64, kernelsize=3, strides=1, paddings=1, bias=True, **self.kwargs)
        self.layers.append(self.conv2)
        self.conv3 = FF_ConvLayer_2D(64, 64, kernelsize=3, strides=1, paddings=1, bias=True, **self.kwargs)
        self.layers.append(self.conv3)

        # Skip Connection for Layer 1 (Identity)
        self.id1 = nn.Identity()

        # Layer 2
        self.conv4 = FF_ConvLayer_2D(64, 64, kernelsize=3, strides=2, paddings=1, bias=True, **self.kwargs)
        self.layers.append(self.conv4)
        self.conv5 = FF_ConvLayer_2D(64, 64, kernelsize=3, strides=1, paddings=1, bias=True, **self.kwargs)
        self.layers.append(self.conv3)

    
    def predict(self, x, num_classes):
        goodness_per_label = []
        for label in range(num_classes):
            label = torch.tensor([label] * x.size(0)).cuda()
            h = concat_y_and_x(x, label, num_classes)
            goodness = []
            out = self.conv1(h)
            identity = self.id1(out)
            #goodness += [out.pow(2).flatten(1).mean(1)]

            # Layer 1 with identity skip connection
            out = self.conv2(out)
            goodness += [out.pow(2).flatten(1).mean(1)]

            # Layer 2 
            out = self.conv3(out)
            goodness += [out.pow(2).flatten(1).mean(1)]
            out = self.conv4(out + identity)
            goodness += [out.pow(2).flatten(1).mean(1)]
            out = self.conv5(out)
            goodness += [out.pow(2).flatten(1).mean(1)]

            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1), goodness_per_label
    
    
    def train(self, x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug=None, x_neg_aug=None):

        h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug = x_pos, x_neg, x_pos_test, x_neg_test, x_pos_aug, x_neg_aug
        logging.info('Training layer 0 ...')
        h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = self.conv1.train(h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug)
        logging.info('Opt. Threshold: %s' % self.conv1.threshold.item())

        if h_pos_aug is not None:
            h_pos_identity, h_neg_identity, h_pos_test_identity, h_neg_test_identity, h_pos_aug_identity, h_neg_aug_identity = self.id1(h_pos), self.id1(h_neg), self.id1(h_pos_test), self.id1(h_neg_test), self.id1(h_pos_aug), self.id1(h_neg_aug)
        else:
            h_pos_identity, h_neg_identity, h_pos_test_identity, h_neg_test_identity = self.id1(h_pos), self.id1(h_neg), self.id1(h_pos_test), self.id1(h_neg_test)
        
        logging.info('Training layer 1 ...')
        h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = self.conv2.train(h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug)
        logging.info('Opt. Threshold: %s' % self.conv2.threshold.item())

        # Training layer 2
        logging.info('Training layer 2 ...')
        h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = self.conv3.train(h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug)

        logging.info('Opt. Threshold: %s' % self.conv3.threshold.item())

        # Training layer 3
        logging.info('Training layer 3 ...')
        if h_pos_aug is not None:
            h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = self.conv4.train(h_pos + h_pos_identity, h_neg + h_neg_identity, h_pos_test + h_pos_test_identity, h_neg_test + h_neg_test_identity, h_pos_aug + h_pos_aug_identity, h_neg_aug + h_neg_aug_identity)
        else:
            h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = self.conv4.train(h_pos + h_pos_identity, h_neg + h_neg_identity, h_pos_test + h_pos_test_identity, h_neg_test + h_neg_test_identity, h_pos_aug, h_neg_aug)
      
        logging.info('Opt. Threshold: %s' % self.conv4.threshold.item())

        h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug, loss_test = self.conv5.train(h_pos, h_neg, h_pos_test, h_neg_test, h_pos_aug, h_neg_aug)

        logging.info('Opt. Threshold: %s' % self.conv5.threshold.item())
        
        return loss_test
    
    def get_norm_vectors(self, x: torch.Tensor):
        norm_vectors = []
        out1 = self.conv1(x)
        identity = self.id1(out1)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3 + identity)
        out5 = self.conv5(out4)
        norm_vectors = [out1, out2, out3, out4, out5]
        return norm_vectors
    
    def save_checkpoint(self, path, name):

        checkpoint = {
            'model_state_dict': [self.layers[i].state_dict() for i in range(len(self.layers))],
            'optimizer_state_dict': [self.layers[i].opt.state_dict() for i in range(len(self.layers))],
        }
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(checkpoint, path + '/' + name + '.pt')

    def load_checkpoint(self, path, name):

        checkpoint = torch.load(path + '/' + name + '.pt')
        
        # Load optimizer states for each layer
        for i, layer in enumerate(checkpoint['optimizer_state_dict']):
            self.layers[i].load_state_dict(checkpoint['model_state_dict'][i])
            self.layers[i].opt.load_state_dict(checkpoint['optimizer_state_dict'][i])

        next =  i+1
        for j in range(0, len(checkpoint['optimizer_state_dict']) - next):
            del self.layers[-1]
            logging.info('Deleted empty layer: %s ' % (next+j))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from source.utils import log_loss
from source.supervision import generate_contrastive_pairs_x
from source.utils import EarlyStopper, augment_positive_samples

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, features, loss, opt, lr, **kwargs):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dimensions = kwargs.get('dimensions', None)

        self.loss_function = kwargs.get('loss_function', None)
        if self.loss_function is None or self.loss_function == 'cross_entropy_loss':
            self.loss = nn.CrossEntropyLoss()
        elif self.loss_function == 'contrastive':
            self.loss = nn.TripletMarginLoss()

        if not isinstance(kwargs.get('kernel_size', 3), list):
            self.list_kernel_size = [kwargs.get('kernel_size', 3)] * (len(features) + 1)
        else:   
            self.list_kernel_size = kwargs.get('kernel_size', None)
        
        if not isinstance(kwargs.get('stride', 2), list):
            self.list_stride = [kwargs.get('stride', 2)] * (len(features) + 1)
        else:
            self.list_stride = kwargs.get('stride', None)
        
        if not isinstance(kwargs.get('padding', 0), list):
            self.list_padding = [kwargs.get('padding', 1)] * (len(features) + 1)
        else:
            self.list_padding = kwargs.get('padding', None)

        self.dims = [input_channels] + features

        for d in range(len(self.dims) - 1):
            self.layers += [nn.Conv2d(self.dims[d], self.dims[d + 1], kernel_size=self.list_kernel_size[d], padding=self.list_padding[d], stride=self.list_stride[d]).cuda()]
            self.layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1).cuda()]
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to handle variable size
        
        # The last hidden_dim value times the output of the adaptive pooling
        self.fc = nn.Linear(features[-1], num_classes)
        self.num_classes = num_classes
        self.opt = opt(self.parameters(), lr=lr)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        x = self.global_avg_pool(x)  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
    
    def predict(self,x):
        outputs = self.forward(x)
        return outputs.argmax(1), outputs
    
    def train(self,x,y, x_test, y_test):
        
        if self.loss_function == 'cross_entropy_loss':
                outputs = self.forward(x)
                loss = self.loss(outputs, y)
        elif self.loss_function == 'contrastive':
            positive, negative = generate_contrastive_pairs_x(x, y, self.num_classes, False)
            positive_test, negative_test = generate_contrastive_pairs_x(x_test, y_test, self.num_classes, False)
            x_pos_aug, x_pos_aug_test = augment_positive_samples(positive, positive_test, self.dimensions)
            anchor, x_pos_aug, negative = self.forward(positive), self.forward(x_pos_aug), self.forward(negative)
            loss = self.loss(anchor, x_pos_aug, negative)

        with torch.no_grad():
            if self.loss_function == 'cross_entropy_loss':
                    outputs = self.forward(x_test)
                    loss_test = self.loss(outputs, y_test)
            elif self.loss_function == 'contrastive':
                anchor, x_pos_aug, negative = self.forward(positive_test), self.forward(x_pos_aug_test), self.forward(negative_test)
                loss_test  = self.loss(anchor, x_pos_aug, negative)

        loss.backward()
        self.opt.step()
        return loss_test.item()
    
    def save_checkpoint(self, path, name):
        torch.save(self.state_dict(), path + '/' + name + '.pt')

    def load_checkpoint(self, path, name):
        self.load_state_dict(torch.load(path + '/' + name + '.pt'))
    

class ResNet18(nn.Module):
    def __init__(self, num_classes, loss_function, optimiser, learning_rate, pretrained=False, **kwargs):
        super().__init__()
        self.model = models.resnet18()
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, num_classes)
        self.loss_function = kwargs.get('loss_function', None)
        if self.loss_function is None or self.loss_function == 'log_loss':
            self.loss = log_loss
        elif self.loss_function == 'contrastive':
            self.loss = nn.TripletMarginLoss()
        self.optimiser = optimiser(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        if x.shape[1] < 3:
            additional_dims = 3 - x.shape[1] 
            x = torch.cat([x]+[x]*additional_dims, 1)
        return self.model(x)
    
    def predict(self,x):
        outputs = self.forward(x)
        return outputs.argmax(1), outputs
    
    def train(self,x,y, x_test, y_test):
        if self.loss_function == 'log_loss':
                outputs = self.forward(x)
                loss = self.loss(outputs, y)
        elif self.loss_function == 'contrastive':
            positive, negative = generate_contrastive_pairs_x(x, y, self.num_classes, False)
            positive_test, negative_test = generate_contrastive_pairs_x(x_test, y_test, self.num_classes, False)
            x_pos_aug, x_pos_aug_test = augment_positive_samples(positive, positive_test, self.dimensions)
            anchor, x_pos_aug, negative = self.forward(positive), self.forward(x_pos_aug), self.forward(negative)
            loss = self.loss(anchor, x_pos_aug, negative)

        with torch.no_grad():
            if self.loss_function == 'log_loss':
                    outputs = self.forward(x_test)
                    loss_test = self.loss(outputs, y)
            elif self.loss_function == 'contrastive':
                anchor, x_pos_aug, negative = self.forward(positive_test), self.forward(x_pos_aug_test), self.forward(negative_test)
                loss_test  = self.loss(anchor, x_pos_aug, negative)

        loss.backward()
        self.optimiser.step()
        return loss_test.item()
    
    def save_checkpoint(self, path, name):
        torch.save(self.state_dict(), path + '/' + name + '.pt')

    def load_checkpoint(self, path, name):
        self.load_state_dict(torch.load(path + '/' + name + '.pt'))


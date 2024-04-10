
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, h_dims, loss, opt, lr):
        super().__init__()

        self.layers = nn.ModuleList()
        self.loss = loss
        self.dims = [input_size] + h_dims
        
        # Create hidden layers dynamically based on hidden_dims
        for d in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[d], self.dims[d + 1]))

        # Output layer
        self.output_layer = nn.Linear(self.dims[-1], num_classes)

        self.opt = opt(self.parameters(), lr=lr)
    
    def forward(self, x):

        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Output layer without ReLU (assuming classification task)
        x = self.output_layer(x)
        return x
    
    def predict(self,x):
        outputs = self.forward(x)
        return outputs.argmax(1), outputs
    
    def train(self,x, y, x_test, y_test):
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        
        with torch.no_grad():
            outputs_test = self.forward(x_test)
            loss_test = self.loss(outputs_test, y_test)

        loss.backward()
        self.opt.step()
        return loss_test.item()
    
    def save_checkpoint(self, path, name):
        torch.save(self.state_dict(), path + '/' + name + '.pt')

    def load_checkpoint(self, path, name):
        self.load_state_dict(torch.load(path + '/' + name + '.pt'))
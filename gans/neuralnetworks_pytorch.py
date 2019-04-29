import numpy as np
import torch
from torch.autograd import Variable
import time
import gzip
import pickle
import matplotlib.pyplot as plt
plt.ion()

# Assumes input images and kernels are square

class NeuralNetworkClassifier_Pytorch(torch.nn.Module):
    def __init__(self, n_inputs, n_hiddens_by_layer, n_outputs,
                 relu=False, gpu=False,
                 n_conv_layers=0, windows=[], strides=[],
                 input_height_width=None):
        if n_conv_layers != len(windows) != len(strides):
            raise(Exception('NeuralNetworkClassifier_Pytorch: n_conv_layers != len(windows) != len(strides):'))
        super().__init__()

        self.activation = torch.relu if relu else torch.tanh
        self.n_outputs = n_outputs

        # Build each convolutional layer
        self.conv_layers = torch.nn.ModuleList()
        for n_units, window, stride in zip(n_hiddens_by_layer[:n_conv_layers],
                                           windows, strides):
            self.conv_layers.append(torch.nn.Conv2d(n_inputs, n_units,
                                                    kernel_size=window,
                                                    stride=stride))
            n_inputs = n_units
            input_height_width = (input_height_width - window) // stride + 1

        # n_inputs to fc layer results from flattening all outputs
        # from previous convolutional layer
        if n_conv_layers > 0:
            n_inputs = input_height_width ** 2 * n_inputs
        self.fc_layers = torch.nn.ModuleList()
        for n_units in n_hiddens_by_layer[n_conv_layers:]:
            self.fc_layers.append(torch.nn.Linear(n_inputs, n_units))
            n_inputs = n_units
        self.fc_layers.append(torch.nn.Linear(n_inputs, n_outputs))

        if gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.to(self.device)

    def forward_all_outputs(self, X):
        n_samples = X.shape[0]
        Ys = [X]
        # print(Ys[-1].shape)
        for i in range(len(self.conv_layers)):
            # print('i', i, 'Input', Ys[-1].shape)
            Ys.append(self.activation(self.conv_layers[i](Ys[-1])))

        for i in range(len(self.fc_layers)-1):
            if i == 0:
                # flatten for input to first fc layer
                Ys.append(self.activation(self.fc_layers[i](Ys[-1].reshape(n_samples, -1))))
            else:
                Ys.append(self.activation(self.fc_layers[i](Ys[-1])))
        Ys.append(self.fc_layers[-1](Ys[-1]))

        return Ys[1:]  # all outputs without original inputs

    def forward(self, X):
        Ys = self.forward_all_outputs(X)
        return Ys[-1]

    def train(self, data_loader, n_iterations,
              learning_rate, verbose=True):

        start_time = time.time()

        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for i in range(n_iterations):
            for i, (imgs, targets) in enumerate(data_loader):
                Xtrain_batch = Variable(imgs.to(self.device), requires_grad=False)
                Ttrain_batch = Variable(targets.to(self.device), requires_grad=False)

                optimizer.zero_grad()
                Y = self.forward(Xtrain_batch)
                output = loss.forward(Y, Ttrain_batch)

                output.backward()
                optimizer.step()

        delta_time = time.time() - start_time
        self.time = delta_time

    def use(self, X):
        X = Variable(X.to(self.device), requires_grad=False)
        Y = self.forward(X).detach().cpu().numpy()
        classes = Y.argmax(axis=1)
        Ye = np.exp(Y)
        probs = Ye / np.sum(Ye, axis=1).reshape((-1, 1))
        return classes, probs, Y

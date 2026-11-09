import numpy as np
import time
class Cnn:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            if not (type(layer).__name__ == 'dropout_layer' and not self.dropout):
                x=layer.forward(x)
            #print(type(layer).__name__, np.abs(x).max())
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if not (type(layer).__name__ == 'dropout_layer' and not self.dropout):
                grad = layer.backward(grad)
        return grad

    def dropout(self, on: bool):
        self.dropout = on
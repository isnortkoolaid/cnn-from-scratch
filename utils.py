import numpy as np  

def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def onehot_encode(y: tuple[int, ...], num_classes):
    return (y[:, np.newaxis] == np.arange(num_classes)).astype(int)

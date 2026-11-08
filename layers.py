import numpy as np
import utils
class convol_layer:
    def __init__(self, 
                 kernel_dims: tuple[int, int, int, int], 
                 img_dims: tuple[int, int, int],
                 learning_rate: float):
        self.kernel_dims = kernel_dims
        self.img_dims = img_dims
        self.kernel = np.random.rand(*kernel_dims)
        self.biases = np.random.rand(kernel_dims[0], img_dims[1] - kernel_dims[2] + 1, img_dims[2] - kernel_dims[3] + 1)
        self.learning_rate = learning_rate
    def forward(self, img):
            self.img = img
            output = utils.convolve(img, self.kernel)+self.biases
            return output
    def backward(self, dL_dout): 
        dL_dkernel = np.zeros(self.kernel_dims)
        dL_dimg = np.zeros(self.img_dims)
        dL_dkernel = utils.convolve(dL_dout, self.img)
        dL_dimg = utils.convolve(dL_dout, np.rot90(self.kernel, 2, axes=(2, 3))) 
        dL_db = np.sum(dL_dout, axis=(1, 2)) 
        self.kernel -= self.learning_rate * dL_dkernel 
        self.biases -= self.learning_rate * dL_db 
        return dL_dimg
class max_pool_layer:
    def __init__(self, pool_size: tuple[int, int]):
        self.pool_size = pool_size
    def forward(self, img):
        self.img = img
        self.indices = np.zeros_like(img, dtype=bool)
        output = np.zeros((img.shape[0], img.shape[1] // self.pool_size[0], img.shape[2] // self.pool_size[1]))
        for d in range(img.shape[0]):
            for i in range(img.shape[1] // self.pool_size[0]):
                for j in range(img.shape[2] // self.pool_size[1]):
                    output[d, i, j] = np.max(img[d, i * self.pool_size[0]: (i + 1) * self.pool_size[0], j * self.pool_size[1]: (j + 1) * self.pool_size[1]])
                    max_index = np.unravel_index(np.argmax(img[d, i * self.pool_size[0]: (i + 1) * self.pool_size[0], j * self.pool_size[1]: (j + 1) * self.pool_size[1]]), self.pool_size)
                    self.indices[d, i * self.pool_size[0] + max_index[0], j * self.pool_size[1] + max_index[1]] = True
        return output
    def backward(self, dL_dout):
        dL_dimg = np.zeros_like(self.img)
        for d in range(self.indices.shape[0]):
            for i in range(self.indices.shape[1]):
                for j in range(self.indices.shape[2]):
                    if self.indices[d, i, j]:
                        dL_dimg[d, i, j] = dL_dout[d, i // self.pool_size[0], j // self.pool_size[1]]
        return dL_dimg
class activation_layer:
    def __init__(self):
        pass
    def forward(self, img):
        self.img = img
        return utils.relu(img)
    def backward(self, dL_dout):
        return dL_dout * (self.img > 0)
class dropout_layer:
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
    def forward(self, img):
        self.mask = np.random.rand(*img.shape) > self.dropout_rate
        return (img * self.mask)*(1/(1-self.dropout_rate))
    def backward(self, dL_dout):
        return dL_dout * self.mask
class flatten_layer:
    def __init__(self):
        pass
    def forward(self, img):
        self.img_shape = img.shape
        return img.flatten()
    def backward(self, dL_dout):
        return dL_dout.reshape(self.img_shape)
class fc_layer:
    def __init__(self, layer_sizes: tuple[int, int, int], learning_rate: float):
        self.weights1 = np.random.rand(layer_sizes[1], layer_sizes[0])
        self.weights2 = np.random.rand(layer_sizes[2], layer_sizes[1])
        self.biases1 = np.random.rand(layer_sizes[1])
        self.biases2 = np.random.rand(layer_sizes[2])
        self.learning_rate = learning_rate
    def forward(self, img):
        self.img = img
        self.hidden = utils.relu(np.dot(self.weights1, img) + self.biases1)
        
        output = utils.relu(np.dot(self.weights2, self.hidden) + self.biases2)
        return output
    def backward(self, dL_dout):
        dL_dweights2 = np.outer(dL_dout, self.hidden)
        dL_dbiases2 = dL_dout
        dL_dz = np.dot(self.weights2.T, dL_dout)
        dL_dweights1 = np.outer(dL_dz, self.img)
        dL_dbiases1 = dL_dz
        dL_dimg = np.dot(self.weights1.T, dL_dz)
        self.weights1 -= self.learning_rate * dL_dweights1
        self.biases1 -= self.learning_rate * dL_dbiases1
        self.weights2 -= self.learning_rate * dL_dweights2
        self.biases2 -= self.learning_rate * dL_dbiases2
        return dL_dimg
class softmax_layer:
    def __init__(self):
        pass
    def forward(self, img):
        shift = img - np.max(img)
        self.output = np.exp(shift) / np.sum(np.exp(shift))
        return self.output
    def backward(self, dL_dout):
        dL_dimg = self.output * (1-self.output) * dL_dout
        return dL_dimg
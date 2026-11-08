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
        dL_dkernel = utils.convolve(self.img, dL_dout, 2, axes=(1, 2))
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
    def __init__(self, input_size: int, output_size: int, learning_rate: float):
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size)
        self.learning_rate = learning_rate
    def forward(self, img):
        self.img = img
        return np.dot(self.weights, img) + self.biases
    def backward(self, dL_dout):
        dL_dweights = np.outer(dL_dout, self.img)
        dL_dbiases = dL_dout
        dL_dimg = np.dot(self.weights.T, dL_dout)
        self.weights -= self.learning_rate * dL_dweights
        self.biases -= self.learning_rate * dL_dbiases
        return dL_dimg
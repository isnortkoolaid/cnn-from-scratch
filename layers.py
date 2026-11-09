import numpy as np
import utils
class convol_layer:
    def __init__(self, 
                 kernel_dims: tuple[int, int, int, int], 
                 img_dims: tuple[int, int, int, int],
                 learning_rate: float):
        self.kernel_dims = kernel_dims
        self.img_dims = img_dims
        self.kernel = np.random.randn(*kernel_dims) * np.sqrt(2/(kernel_dims[2]*kernel_dims[3]*kernel_dims[1]))
        self.biases = np.zeros((1, kernel_dims[0], 1, 1))
        self.learning_rate = learning_rate
    def forward(self, img):
        self.img = img
        self.output = np.zeros((self.img_dims[0], self.kernel_dims[0], self.img_dims[2] - self.kernel_dims[2] + 1, self.img_dims[3] - self.kernel_dims[3] + 1))
        for i in range(self.kernel_dims[2]):
            for j in range(self.kernel_dims[3]):
                #cool way to convolve
                patch=img[:, :, i:i + self.img_dims[2] - self.kernel_dims[2] + 1, j:j + self.img_dims[3] - self.kernel_dims[3] + 1]
                self.output += np.tensordot(patch, self.kernel[:, :, i, j], axes=([1], [1])).transpose(0,3,1,2) # transpose cus output is in (img, rows, cols, output channel)
        return self.output+self.biases
    def backward(self, dL_dout):
        dL_dkernel = np.zeros(self.kernel_dims)
        dL_dimg = np.zeros(self.img_dims)
        for i in range(self.kernel_dims[2]):
            for j in range(self.kernel_dims[3]):
                patch=self.img[:, :, i:i + self.img_dims[2] - self.kernel_dims[2] + 1, j:j + self.img_dims[3] - self.kernel_dims[3] + 1]
                #multiply and sum across batch, rows, cols
                dL_dkernel[:,:,i,j] = np.tensordot(dL_dout, patch, axes=([0,2,3],[0,2,3]))
                #dL_d(patch) is sum of products between dl_dout and kernel at the area corresponding to patch across output channels
                dL_dimg[:,:,i:i + self.img_dims[2] - self.kernel_dims[2] + 1, j:j + self.img_dims[3] - self.kernel_dims[3] + 1] += np.tensordot(
                    dL_dout, self.kernel[:,:,i,j], axes=([1],[0])).transpose(0,3,1,2) # transpose cus output is in (img, rows, cols, output channels)
        dL_db = np.sum(dL_dout, axis=(0,2,3)).reshape(-1,1,1)
        self.kernel -= self.learning_rate*dL_dkernel / self.img_dims[0]
        self.biases -= self.learning_rate*dL_db / self.img_dims[0]
        return dL_dimg
class max_pool_layer:
    def __init__(self, pool_size: tuple[int, int]):
        self.pool_size = pool_size
    def forward(self, img):
        self.img = img
        self.indices = np.zeros_like(img, dtype=bool)
        #splits the input into blocks of size N x D x Ho x Ph x Wo x Pw, so there are N x D x Ho x Wo blocks
        blocks = img[:, :, :self.img.shape[2], :self.img.shape[3]].reshape(self.img.shape[0], self.img.shape[1],(self.img.shape[2] // self.pool_size[0]), self.pool_size[0],(self.img.shape[3] // self.pool_size[1]), self.pool_size[1])
        output = blocks.max(axis=(3,5))
        self.mask = (blocks==output[:,:,:,None,:,None])
        return output
    def backward(self, dL_dout):
        dL_dimg = np.zeros(self.img.shape)
        dL_dimg[:,:,:self.img.shape[2], :self.img.shape[3]] = (self.mask * dL_dout[:,:,:,None,:,None]).reshape(self.img.shape)
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
        return dL_dout * self.mask / (1 - self.dropout_rate)
class flatten_layer:
    def __init__(self):
        pass
    def forward(self, img):
        self.img_shape = img.shape
        return img.reshape(img.shape[0], -1)
    def backward(self, dL_dout):
        return dL_dout.reshape(self.img_shape)
class fc_layer:
    def __init__(self, layer_sizes: tuple[int, int, int], batch_size: int, learning_rate: float):
        self.weights1 = np.random.randn(layer_sizes[0], layer_sizes[1]) * np.sqrt(2/layer_sizes[0])
        self.weights2 = np.random.randn(layer_sizes[1], layer_sizes[2]) * np.sqrt(2/layer_sizes[1])
        self.biases1 = np.zeros(layer_sizes[1])
        self.biases2 = np.zeros(layer_sizes[2])
        self.hidden = np.zeros((batch_size, layer_sizes[1]))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    def forward(self, img):
        self.img = img
        output = np.zeros((self.batch_size, self.weights2.shape[1]))
        for i in range(self.batch_size):
            self.hidden[i] = utils.relu( self.img[i]@self.weights1 + self.biases1)
            output[i] = self.hidden[i] @ self.weights2 + self.biases2
        return output
    def backward(self, dL_dout):
        dL_dweights2 = self.hidden.T @ dL_dout
        dL_dbiases2 = np.sum(dL_dout, axis=0)
        dL_dhidden = dL_dout @ self.weights2.T * (self.hidden > 0)
        dL_dhidden[self.hidden <= 0] = 0
        dL_dweights1 = self.img.T @ dL_dhidden
        dL_dbiases1 = np.sum(dL_dhidden, axis=0)
        dL_dimg = dL_dhidden @ self.weights1.T
        self.weights1 -= self.learning_rate * dL_dweights1 / self.batch_size
        self.biases1 -= self.learning_rate * dL_dbiases1 / self.batch_size
        self.weights2 -= self.learning_rate * dL_dweights2 / self.batch_size
        self.biases2 -= self.learning_rate * dL_dbiases2 / self.batch_size
        return dL_dimg
class softmax_layer:
    def __init__(self):
        pass
    def forward(self, img):
        self.output = np.zeros_like(img)
        for i  in range(img.shape[0]): #for each img in batch
            shift = img[i] - np.max(img[i])
            self.output[i] = np.exp(shift) / np.sum(np.exp(shift))
        return self.output
    
    def backward(self, dL_dout):
        return dL_dout
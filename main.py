import utils
import layers
import cnn
import numpy as np
from tensorflow.keras.datasets import mnist

network = cnn.Cnn([
    layers.convol_layer(kernel_dims=(32, 1, 5, 5),
                        img_dims=(1, 28, 28),
                        learning_rate=0.01),
    layers.activation_layer(),
    layers.max_pool_layer(pool_size=(2, 2)),
    layers.convol_layer(kernel_dims=(64, 32, 3, 3),
                        img_dims=(32, 12, 12),
                        learning_rate=0.01),
    layers.activation_layer(),
    layers.max_pool_layer(pool_size=(2, 2)),
    layers.flatten_layer(),
    layers.dropout_layer(dropout_rate=0.5),
    layers.fc_layer(layer_sizes=(64*5*5, 128, 10), learning_rate=0.01),
    layers.softmax_layer()])
(x_train, y_train), (x_test, y_test) = mnist.load_data();
x_train = x_train.reshape(-1, 1, 28, 28) / 255.0
x_test = x_test.reshape(-1, 1, 28, 28) / 255.0
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        output = network.forward(x)
        loss = -np.log(output[y])
        dL_dout = np.zeros_like(output)
        dL_dout[y] = -1 / output[y]
        network.backward(dL_dout)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(x_train)} samples, Loss: {loss:.4f}")
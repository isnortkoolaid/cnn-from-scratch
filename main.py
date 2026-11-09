import utils
import layers
import cnn
import numpy as np
from tensorflow.keras.datasets import mnist

network = cnn.Cnn([
    layers.convol_layer(kernel_dims=(32, 1, 5, 5),
                        img_dims=(32, 1, 28, 28),
                        learning_rate=0.01),
    layers.activation_layer(),
    layers.max_pool_layer(pool_size=(2, 2)),
    layers.convol_layer(kernel_dims=(64, 32, 3, 3),
                        img_dims=(32, 32, 12, 12),
                        learning_rate=0.01),
    layers.activation_layer(),
    layers.max_pool_layer(pool_size=(2, 2)),
    layers.flatten_layer(),
    layers.dropout_layer(dropout_rate=0.5),
    layers.fc_layer(layer_sizes=(64*5*5, 128, 10), batch_size=32, learning_rate=0.01),
    layers.softmax_layer()])
(x_train, y_train), (x_test, y_test) = mnist.load_data();
x_train = x_train.reshape(-1, 1, 28, 28) / 255.0
x_test = x_test.reshape(-1, 1, 28, 28) / 255.0
epochs = 10
loss=0
batch_size = 32
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    for i in range(0, len(x_train), batch_size):
        x = x_train[i:i+batch_size]
        y = utils.onehot_encode(y_train[i:i+batch_size], 10)
        output = network.forward(x)
        loss += -np.sum(y * np.log(output + 1e-8))
        dL_dout = output - y
        network.backward(dL_dout)
        if (i+batch_size) % 1024 == 0:
            print(f"Processed {i+batch_size}/{len(x_train)} samples, Loss: {loss/(1024*batch_size)}")
            loss=0
correct = 0;
network.dropout(False)
for i in range(0, len(x_test)-16, batch_size):
    x = x_test[i:i+batch_size]
    y = y_test[i:i+batch_size]
    output = network.forward(x)
    correct+= np.count_nonzero(np.argmax(output, axis=1) == y)
print(f"Score: {correct/len(x_test)}")
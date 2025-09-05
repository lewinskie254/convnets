import h5py
import numpy as np
from convolution import ConvLayer
from pooling import PoolLayer
from linear import FCLayer  # your new simplified class

# Load dataset
file1 = h5py.File(r"C:\Users\dell\Desktop\pytorch\train_catvnoncat.h5", 'r')
file2 = h5py.File(r"C:\Users\dell\Desktop\pytorch\test_catvsnoncat.h5", 'r')

def load_dataset(train_dataset, test_dataset):
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])
    return train_set_x, train_set_y, test_set_x, test_set_y

X_train, Y_train, X_test, Y_test = load_dataset(file1, file2)

# Preprocess
X_train = X_train / 255.0
X_test = X_test / 255.0
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# Example dimensions
conv_filter = 'horizontal'
pool_size = 2
stride_conv = 1
stride_pool = 2
lr = 0.01

# Sample image
sample_image = X_train[0].reshape(64,64,3)

# Initialize Conv layer
conv = ConvLayer(sample_image, filter=conv_filter, stride=stride_conv, padding=0, lr=lr)

# Forward pass through Conv → Pool to get flattened size
conv_out = conv.forward(sample_image)
pool = PoolLayer(conv_out, pool_size=pool_size, stride=stride_pool)
pool_out = pool.forward()
flattened_size = pool_out.flatten().shape[0]

# Initialize FCLayer
fc = FCLayer(input_dim=flattened_size, output_dim=1, activation="sigmoid", learning_rate=lr)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for i in range(X_train.shape[0]):
        img = X_train[i].reshape(64,64,3)
        label = Y_train[i:i+1]

        # Forward pass
        conv_out = conv.forward(img)
        pool_out = pool.forward()
        flattened = pool_out.flatten()[np.newaxis, :]

        fc_out = fc.forward(flattened)

        # Compute simple loss (binary cross-entropy)
        epsilon = 1e-15
        yhat = np.clip(fc_out, epsilon, 1 - epsilon)
        loss = -np.mean(label * np.log(yhat) + (1 - label) * np.log(1 - yhat))
        total_loss += loss

        # Backward pass
        d_loss = (fc_out - label)  # derivative of BCE w.r.t output
        d_fc = fc.backward(d_loss)

        # Backprop through Pool → Conv
        d_pool = pool.backward(d_fc.reshape(pool_out.shape))
        conv.backward(d_pool)

    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/X_train.shape[0]:.6f}")

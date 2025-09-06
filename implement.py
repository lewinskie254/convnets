import h5py
import numpy as np
from convolution import ConvLayer
from pooling import PoolLayer
from linear import FCLayer

# ---------------------------
# Dataset Loading & Preprocessing
# ---------------------------
def load_dataset(train_file, test_file):
    train_set_x = np.array(train_file["train_set_x"][:])
    train_set_y = np.array(train_file["train_set_y"][:])
    test_set_x = np.array(test_file["test_set_x"][:])
    test_set_y = np.array(test_file["test_set_y"][:])
    return train_set_x, train_set_y, test_set_x, test_set_y

# Load files
file1 = h5py.File(r"C:\Users\dell\Desktop\pytorch\train_catvnoncat.h5", 'r')
file2 = h5py.File(r"C:\Users\dell\Desktop\pytorch\test_catvsnoncat.h5", 'r')

X_train, Y_train, X_test, Y_test = load_dataset(file1, file2)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)




# ---------------------------
# Training Loop
# ---------------------------
conv = ConvLayer(filter='horizontal', lr=0.001)
pool = PoolLayer(pool_size=2, stride=2)
fc = FCLayer(input_dim=31*31, output_dim=1, activation='sigmoid', lr=0.01)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for i in range(X_train.shape[0]):
        img = X_train[i].reshape(64,64,3)
        label = Y_train[i:i+1]

        # Forward
        conv_out = conv.forward(img)
        pool_out = pool.forward(conv_out)
        flat = pool_out.flatten()[np.newaxis, :]
        fc_out = fc.forward(flat)

        # Binary cross-entropy loss
        loss = - (label*np.log(fc_out+1e-15) + (1-label)*np.log(1-fc_out+1e-15))
        total_loss += loss

        # Backward
        d_fc = fc_out - label
        d_flat = fc.backward(d_fc)
        d_pool = d_flat.reshape(pool_out.shape)
        d_conv = pool.backward(d_pool)
        conv.backward(d_conv)
    
    print(f"Epoch {epoch+1}, Avg Loss: {float(total_loss) / X_train.shape[0]:.6f}")



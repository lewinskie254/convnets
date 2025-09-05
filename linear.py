import numpy as np

class FCLayer:
    def __init__(self, input_dim, output_dim, activation="relu", learning_rate=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.learning_rate = learning_rate

        # Weight initialization
        if self.activation_name == "relu":
            initializer = np.sqrt(2 / input_dim)
        else:
            initializer = np.sqrt(1 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * initializer
        self.b = np.zeros((1, output_dim))

    # --- Activations ---
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x): 
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x): return np.tanh(x)
    def tanh_derivative(self, x): return 1 - np.tanh(x) ** 2

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)

    def activate(self, x):
        if self.activation_name == "sigmoid": return self.sigmoid(x)
        if self.activation_name == "tanh": return self.tanh(x)
        if self.activation_name == "relu": return self.relu(x)

    def activate_derivative(self, x):
        if self.activation_name == "sigmoid": return self.sigmoid_derivative(x)
        if self.activation_name == "tanh": return self.tanh_derivative(x)
        if self.activation_name == "relu": return self.relu_derivative(x)

    # --- Forward pass ---
    def forward(self, x):
        self.x = x                           # cache input
        self.z = np.dot(x, self.W) + self.b  # linear step
        self.a = self.activate(self.z)       # activation step
        return self.a

    # --- Backward pass ---
    def backward(self, d_output):
        # d_output is gradient of loss wrt this layer’s output (a)
        dz = d_output * self.activate_derivative(self.z)    # chain rule
        dw = np.dot(self.x.T, dz) / self.x.shape[0]
        db = np.sum(dz, axis=0, keepdims=True) / self.x.shape[0]
        d_input = np.dot(dz, self.W.T)                      # gradient for previous layer

        # Update weights
        self.W -= self.learning_rate * dw
        self.b -= self.learning_rate * db

        return d_input

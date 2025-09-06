import numpy as np

class FCLayer:
    def __init__(self, input_dim, output_dim, activation='relu', lr=0.01):
        self.lr = lr
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation_name = activation.lower()
    
    def activate(self, x):
        if self.activation_name == 'relu': return np.maximum(0, x)
        if self.activation_name == 'sigmoid': return 1/(1+np.exp(-x))
    def activate_derivative(self, x):
        if self.activation_name == 'relu': return (x>0).astype(float)
        if self.activation_name == 'sigmoid': return self.activate(x)*(1-self.activate(x))

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activate(self.z)
        return self.a

    def backward(self, d_output):
        dz = d_output * self.activate_derivative(self.z)
        dW = np.dot(self.x.T, dz)/self.x.shape[0]
        db = np.sum(dz, axis=0, keepdims=True)/self.x.shape[0]
        d_input = np.dot(dz, self.W.T)
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return d_input
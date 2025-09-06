import numpy as np 

class PoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        h, w, c = x.shape
        out_h = (h - self.pool_size)//self.stride + 1
        out_w = (w - self.pool_size)//self.stride + 1
        self.out = np.zeros((out_h, out_w, c))
        self.max_indices = np.zeros_like(x, dtype=bool)
        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, ch]
                    self.out[i,j,ch] = np.max(region)
                    self.max_indices[i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, ch] = (region == np.max(region))
        return self.out

    def backward(self, d_output):
        d_input = np.zeros_like(self.x)
        for ch in range(self.x.shape[2]):
            for i in range(d_output.shape[0]):
                for j in range(d_output.shape[1]):
                    d_input[i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, ch] += d_output[i,j,ch] * self.max_indices[i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, ch]
        return d_input

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
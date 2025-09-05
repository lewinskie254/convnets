import numpy as np 
import pickle 

class FCLayer:
    def __init__(self, inputs, outputs, layers, activation, bias, learning_rate=None, loss=None,
             regularization=None, lambda_reg=0.01):
        self.layers = layers
        self.loss = loss
        self.outputs = outputs 
        self.activation = activation.lower()
        self.learning_rate = learning_rate
        self.inputs = inputs 
        self.bias = bias
        self.weights = {}
        self.biases = {}
        self.z_values = {}
        self.activation_layers = {}
        self.dz = {}
        self.dw = {}
        self.db = {}
        self.regularization = regularization  # "l1", "l2", or None
        self.lambda_reg = lambda_reg
    #Sigmoid Activation 
    def sigmoid(self, x):
        return 1/(1+(np.exp(-x))) 

    #For Classification Purposes 
    def sigmoid_derivative(self, x): 
        sigmoid_val = self.sigmoid(x)
        return sigmoid_val * (1 - sigmoid_val)
    

    #Tanh Activation 
    def tan_h(self, x): 
        numerator = np.exp(x) - np.exp(-x)
        denominator = np.exp(x) + np.exp(-x)
        return numerator / denominator
    
    #Tan H derivative 
    def tan_h_derivative(self, x): 
        tan_h_val = self.tan_h(x)
        derivative = 1 - tan_h_val**2 
        return derivative 
    
    #ReLu 
    def reLU(self, x): 
        return np.maximum(0, x)
    
    #Derivative of Relu 
    def reLU_derivative(self, x): 
        return np.where(x > 0, 1, 0)
    
    #MultiClass Softmax 
    def soft_max(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp_x

    #Initialize Weights 
    def initialize_weights(self, input_dim, output_dim):
        if self.activation == 'relu':
            initializer = np.sqrt(2 / input_dim)
        else:
            initializer = np.sqrt(1 / input_dim)
        return np.random.randn(input_dim, output_dim) * initializer


    #Determine which activation to use 
    def determine_activation(self, x): 
        if self.activation == "sigmoid":
            return self.sigmoid(x)
        elif self.activation == "tanh":
            return self.tan_h(x)
        elif self.activation == "relu":
            return self.reLU(x)


    def hidden_layer_weights(self, index): 
        w = self.initialize_weights(input_dim=self.layers[index-1], output_dim=self.layers[index])
        return w


    def forward(self):
        a = self.inputs
        self.activation_layers[0] = a

        for i in range(len(self.layers)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values[i] = z
            a = self.determine_activation(z)
            self.activation_layers[i + 1] = a

        return a


    #initialize Parameters 
    def initialize_parameters(self):
        a = self.inputs
        input_dim = a.shape[1]

        for i in range(len(self.layers)):
            output_dim = self.layers[i]
            self.weights[i] = self.initialize_weights(input_dim, output_dim)
            self.biases[i] = np.zeros((1, output_dim))
            input_dim = output_dim
    
    #Regularization Penalty 
    def regularization_penalty(self):
        penalty = 0
        if self.regularization == "l2":
            for w in self.weights.values():
                penalty += np.sum(np.square(w))
        elif self.regularization == "l1":
            for w in self.weights.values():
                penalty += np.sum(np.abs(w))
        return self.lambda_reg * penalty


    #Loss Functions 

    def cross_entropy_loss(self, yhat, y):
        epsilon = 1e-15
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return loss + self.regularization_penalty()

    def l1_loss(self, yhat, y):
        loss = np.mean(np.abs(y - yhat))
        return loss + self.regularization_penalty()

    def l2_loss(self, yhat, y):
        loss = np.mean(np.square(y - yhat))
        return loss + self.regularization_penalty()


    def back_propagate(self):
        y = self.outputs
        yhat = self.activation_layers[len(self.layers)]  # last activation
        m = y.shape[0]  # number of samples

        # Choose loss function
        if self.loss == "l2":
            loss = self.l2_loss(yhat, y)
        elif self.loss == "l1":
            loss = self.l1_loss(yhat, y)
        else:
            loss = self.cross_entropy_loss(yhat, y)  # default

        # --- Output layer gradient ---
        L = len(self.layers) - 1
        self.dz[L] = yhat - y  # assuming sigmoid + cross-entropy

        # Loop through layers in reverse
        for l in reversed(range(L + 1)):
            a_prev = self.activation_layers[l]
            self.dw[l] = np.dot(a_prev.T, self.dz[l]) / m
            self.db[l] = np.sum(self.dz[l], axis=0, keepdims=True) / m

            if l != 0:
                z_prev = self.z_values[l - 1]
                if self.activation == "sigmoid":
                    da_prev = np.dot(self.dz[l], self.weights[l].T)
                    self.dz[l - 1] = da_prev * self.sigmoid_derivative(z_prev)
                elif self.activation == "tanh":
                    da_prev = np.dot(self.dz[l], self.weights[l].T)
                    self.dz[l - 1] = da_prev * self.tan_h_derivative(z_prev)
                elif self.activation == "relu":
                    da_prev = np.dot(self.dz[l], self.weights[l].T)
                    self.dz[l - 1] = da_prev * self.reLU_derivative(z_prev)

        # Update weights and biases
        for l in range(L + 1):
            self.weights[l] -= self.learning_rate * self.dw[l]
            self.biases[l] -= self.learning_rate * self.db[l]

        return loss
    
    #train the neural network
    def train(self, epochs, verbose=True):
        for epoch in range(epochs):
            self.forward()            # run forward pass once
            loss = self.back_propagate()  # then backprop on stored activations

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.6f}")

    
    def predict(self, X=None):
        if X is not None:
            self.inputs = X
        yhat = self.forward()
        predictions = (yhat > 0.5).astype(int)
        return predictions

    def evaluate_accuracy(self, X, y_true):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y_true) * 100
        return accuracy


    #save the model 
    def save_model(self, filename="model_weights.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases
            }, f)
    

    #Load the model 
    def load_model(self, filename="model_weights.pkl"):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
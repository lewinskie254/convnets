class PoolLayer:
    def forward(self, input):
        ...
        # max pooling or average pooling
    def backward(self, d_output):
        # propagate gradients
        pass 

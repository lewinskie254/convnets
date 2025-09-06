import numpy as np 
class ConvLayer:
    def __init__(self, filter, stride=1, padding=0, lr=0.01):
        self.stride = stride
        self.padding = padding
        self.lr = lr
        # simple edge filter
        self.filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    
    def forward(self, x):
        self.x = x.astype(float)
        h, w, c = x.shape
        f_h, f_w = self.filter.shape
        out_h = (h - f_h + 2*self.padding)//self.stride + 1
        out_w = (w - f_w + 2*self.padding)//self.stride + 1
        self.out = np.zeros((out_h, out_w))
        for i in range(0, out_h*self.stride, self.stride):
            for j in range(0, out_w*self.stride, self.stride):
                region = self.x[i:i+f_h, j:j+f_w, :]
                self.out[i//self.stride, j//self.stride] = np.sum(region * self.filter[:,:,np.newaxis])
        return self.out[..., np.newaxis]

    def backward(self, d_output):
        # simple numeric cast fix
        d_output = d_output.astype(float)
        d_filter = np.zeros_like(self.filter)
        h, w, c = self.x.shape
        f_h, f_w = self.filter.shape
        for i in range(d_output.shape[0]):
            for j in range(d_output.shape[1]):
                region = self.x[i*self.stride:i*self.stride+f_h, j*self.stride:j*self.stride+f_w, :]
                d_filter += np.sum(region * d_output[i,j], axis=2)
        # update filter
        self.filter -= self.lr * d_filter
        return d_output  # for simplicity, pass gradient backward

import numpy as np 

class PoolLayer:
    def __init__(self, convolved_image, pool_size=2, stride=2):
        self.image = convolved_image 
        self.image_x, self.image_y, self.channels = convolved_image.shape
        self.pool_size = pool_size
        self.stride = stride

    def forward(self):
        # Output dimensions
        out_x = (self.image_x - self.pool_size) // self.stride + 1
        out_y = (self.image_y - self.pool_size) // self.stride + 1
        output = np.zeros((out_x, out_y, self.channels))

        for c in range(self.channels):
            for i in range(0, self.image_x - self.pool_size + 1, self.stride): 
                for j in range(0, self.image_y - self.pool_size + 1, self.stride): 
                    region = self.image[i:i+self.pool_size, j:j+self.pool_size, c]
                    output[i//self.stride, j//self.stride, c] = np.max(region)

        return output

    def backward(self, d_output):
        # Not yet implemented
        pass

import numpy as np 

class PoolLayer:
    def __init__(self, convolved_image, pool_size=2, stride=2):
        self.image = convolved_image 
        self.image_x, self.image_y, self.channels = convolved_image.shape
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None   # to remember max positions during forward

    def forward(self):
        # Output dimensions
        out_x = (self.image_x - self.pool_size) // self.stride + 1
        out_y = (self.image_y - self.pool_size) // self.stride + 1
        output = np.zeros((out_x, out_y, self.channels))
        
        # mask for storing max positions
        self.mask = np.zeros_like(self.image)

        for c in range(self.channels):
            for i in range(0, self.image_x - self.pool_size + 1, self.stride): 
                for j in range(0, self.image_y - self.pool_size + 1, self.stride): 
                    region = self.image[i:i+self.pool_size, j:j+self.pool_size, c]
                    max_val = np.max(region)
                    output[i//self.stride, j//self.stride, c] = max_val

                    # store mask (1 at max location, 0 elsewhere)
                    max_mask = (region == max_val)
                    self.mask[i:i+self.pool_size, j:j+self.pool_size, c] = max_mask

        return output

    def backward(self, d_output):
        """
        d_output: gradient from the next layer, same shape as forward output
        returns: gradient wrt input image (same shape as self.image)
        """
        d_input = np.zeros_like(self.image)

        for c in range(self.channels):
            for i in range(0, self.image_x - self.pool_size + 1, self.stride): 
                for j in range(0, self.image_y - self.pool_size + 1, self.stride): 
                    # upstream gradient
                    grad = d_output[i//self.stride, j//self.stride, c]

                    # route gradient to the max position(s)
                    region = self.mask[i:i+self.pool_size, j:j+self.pool_size, c]
                    d_input[i:i+self.pool_size, j:j+self.pool_size, c] += grad * region

        return d_input

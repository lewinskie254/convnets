import numpy as np 

class ConvLayer: 
    def __init__(self, image, filter='horizontal', stride=1, padding=0):
        self.input_dim_x = image.shape[0]   # height
        self.input_dim_y = image.shape[1]   # width
        self.no_of_channels = image.shape[2]  # channels

        #Filters to use 
        self.horizontal_edges_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) 
        self.vertical_edges_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) 
        self.sharpen_edges_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1,  0]]) 
        self.emboss_filter = np.array([[-2, -1,  0], [-1,  1,  1], [ 0, 1, 2]])
        self.gaussian_blur_filter = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]]) / 16
        self.box_blur_filter = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]) / 9

        self.filter_dim_x = self.horizontal_edges_filter.shape[0]
        self.filter_dim_y = self.horizontal_edges_filter.shape[1]
        self.stride = stride
        self.padding = padding 
        self.image = image 

        match filter: 
            case 'horizontal': 
                self.filter = self.horizontal_edges_filter 
            case 'vertical': 
                self.filter =self.vertical_edges_filter 
            case 'sharpen' :
                self.filter = self.sharpen_edges_filter 
            case 'emboss' :
                self.filter = self.emboss_filter 
            case 'gaussian' : 
                self.filter = self.gaussian_blur_filter 
            case _: 
                self.filter =self.horizontal_edges_filter 
            
            

    def add_padding(self, image, padding, mode='constant', constant_values=0): 
        return np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode=mode, constant_values=constant_values)

        

    def forward(self, x):
        image = self.add_padding(x, self.padding)  # padded image

        # 3D filter spanning all channels
        filter_to_use = np.stack([self.filter]*self.no_of_channels, axis=-1)
        # shape now: (fh, fw, C)

        filter_height, filter_width, _ = filter_to_use.shape
        out_height = (self.input_dim_x + 2 * self.padding - filter_height) // self.stride + 1
        out_width  = (self.input_dim_y + 2 * self.padding - filter_width) // self.stride + 1

        convolved = np.zeros((out_height, out_width))

        for i in range(0, out_height * self.stride, self.stride):
            for j in range(0, out_width * self.stride, self.stride):
                region = image[i:i+filter_height, j:j+filter_width, :]
                convolved[i//self.stride, j//self.stride] = np.sum(region * filter_to_use)


        return convolved


        
                        


    def backward(self): 
        pass 



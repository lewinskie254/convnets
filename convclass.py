import numpy as np 

class ConvLayer: 
    def __init__(self, image, horizontal_filter = None, vertical_filter = None, padding=0):
        self.input_dim_x = image.shape[2] if image.shape[0] > image.shape[1] else image.shape[0]
        self.input_dim_y = image.shape[1]
        self.no_of_channels = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[2]
        self.padding = 0 
        self.horizontal_filter = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) if horizontal_filter == None else horizontal_filter 
        self.vertical_edges_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) if vertical_filter == None else vertical_filter 
        self.filter_dim_x = self.horizontal_filter.shape[0]
        self.filter_dim_y = self.horizontal_filter.shape[1]

    def forward(self): 
        image_dimensions = np.array((self.input_dim_x, self.input_dim_y, self.no_of_channels))
        pass 


    def backward(self): 
        pass 



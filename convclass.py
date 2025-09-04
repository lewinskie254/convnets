import numpy as np 

class ConvLayer: 
    def __init__(self, image, horizontal_filter = None, vertical_filter = None, stride=1, padding=0):
        self.input_dim_x = image.shape[0]   # height
        self.input_dim_y = image.shape[1]   # width
        self.no_of_channels = image.shape[2]  # channels
        self.horizontal_edges_filter = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) if horizontal_filter == None else horizontal_filter 
        self.vertical_edges_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) if vertical_filter == None else vertical_filter
        self.sharpen_edges_filter = np.array([[0, -1,0], [-1, 5, -1], [0, -1,  0]]) 
        self.filter_dim_x = self.horizontal_edges_filter.shape[0]
        self.filter_dim_y = self.horizontal_edges_filter.shape[1]
        self.stride = stride
        self.padding = padding 
        self.image = image 

    def add_padding(self, image, padding, mode='constant', constant_values=0): 
        return np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode=mode, constant_values=constant_values)

        

    def forward(self, x): 
        image = self.add_padding(x, self.padding)
        convolution_filters = [self.horizontal_edges_filter, self.vertical_edges_filter, self.sharpen_edges_filter] 
        convolved_results = [[]]
        for i in range(self.no_of_channels): 
            filter_to_use = convolution_filters[i]
            for j in range(self.input_dim_x): 
                for k in range(self.input_dim_y): 
                    if (j + filter_to_use.shape[0]) < self.input_dim_x and (k + filter_to_use.shape[1] < self.input_dim_y):
                        sum = 0
                        first_row = [image[j][k+0] * filter_to_use[0][0], image[j+1][k+0] * filter_to_use[1][0],image[j+2][k+0] * filter_to_use[2][0]]
                        second_row = [image[j][k+1] * filter_to_use[0][1], image[j+1][k+1] * filter_to_use[1][1],image[j+1][k+1] * filter_to_use[2][1]]
                        third_row = [image[j][k+2] * filter_to_use[0][2], image[j+2][k+2] * filter_to_use[0][0],image[j+2][k+2] * filter_to_use[2][2]]
                        sum += np.sum(first_row) + np.sum(second_row) + np.sum(third_row)
                    convolved_results[k].append(sum)
                    j += 1
                    if j + filter_to_use.shape[0] > self.input_dim_x:
                        k += 1
                    if k + filter_to_use.shape[1] > self.input_dim_y:
                        break 
        return convolved_results 
        
                        


    def backward(self): 
        pass 



# Modified from https://github.com/feiyuhug/lenet-5

from numpy import *

# Build a Layer with [Size x Size]x Numbers neurons
class Layer(object) :
    def __init__(self, lay_size = []) :
        self.lay_size = lay_size
        # maps : feature maps in conv layer and pooling layer
        # For example : in conv1 maps.shape (6, 28, 28)
        #               in fc6 maps.shape (1, 1, 120)
        self.maps = []
        for map_size in lay_size :
            self.maps.append(zeros(map_size))
        self.maps = array(self.maps)
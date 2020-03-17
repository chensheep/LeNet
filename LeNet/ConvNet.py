from numpy import *
from CovLayer import *
from PoolingLayer import *
from FullyConLayer import *
from OutLayer import *

class CovNet(object) :
    def __init__(self) :

        # Define convolution core size of Conv3
        # [[3, 5, 5]] * 6 => [[channel, width, height]] * number_of_convolution_core
        # By the above, "[[4, 5, 5]] * 9" means that we'll have 9 convolution core 
        # and size is [4, 5, 5] (channel, width, height).
        # In Conv3, there are 16 convolution core. See figure 1,2.
        cov3_core_sizes = [[3, 5, 5]] * 6
        cov3_core_sizes.extend([[4, 5, 5]] * 9)
        cov3_core_sizes.extend([[6, 5, 5]])

        # In cov3_mapcombindex, there are 16 lists. 
        # [0,1,2] means that first convolution core operate on channel 0,1 and 2 
        # in previous layer and so on. See figure 1,2.
        cov3_mapcombindex = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],\
                [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],[0,1,3,4],[1,2,4,5],[0,2,3,5],[0,1,2,3,4,5]]

        # See figure 1.
        # Create Conv1
        # [[28, 28]] * 6 = [[28, 28], [28, 28], [28, 28], [28, 28], [28, 28], [28, 28]]
        # In this layer, output is 6 feature maps, and each one size is 28 * 28.
        # [[1, 5, 5]] * 6 = [[1, 5, 5], [1, 5, 5], [1, 5, 5], [1, 5, 5], [1, 5, 5], [1, 5, 5]]
        # In this layer, there are 6 convolution cores, and each one size is 1 * 5 * 5 (channel * width * height).
        self.covlay1 = CovLayer([[28, 28]] * 6, [[1, 5, 5]] * 6)
        
        # Create Pool2
        # [[14, 14]] * 6 : In this layer, output is 6 feature maps, and each one size is 14 * 14.
        # [[2, 2]] * 6   : In this layer, there are 6 pooling cores, and each one size is 2 * 2 (width * height).
        self.poolinglay2 = PoolingLayer([[14, 14]] * 6, [[2, 2]] * 6)
        
        self.covlay3 = CovLayer([[10, 10]] * 16, cov3_core_sizes, cov3_mapcombindex)
        self.poolinglay4 = PoolingLayer([[5, 5]] * 16, [[2, 2]] * 16)
        
        self.covlay5 = CovLayer([[1, 1]] * 120, [[16, 5, 5]] * 120)
        
        # Create FcLayer6
        # Input 120 channels, outpur 84 channels.
        self.fclay6 = FcLayer(84, 120)
        
        # Create OutputLayer7
        # Input 84 channels, outpur 10 channels.
        self.outputlay7 = OutputLayer(10, 84)

    # Forward Propagation
    # We train network one picture each time. So shape of mapset (1, 32, 32) (channel, width, height).
    def fw_prop(self, mapset, mapclass = -1) :
        
        # caculate feature map
        self.covlay1.calc_maps(mapset)
        # Take output of covlay1 as input of poolinglay2
        self.poolinglay2.calc_maps(self.covlay1.maps)
        self.covlay3.calc_maps(self.poolinglay2.maps, True)
        self.poolinglay4.calc_maps(self.covlay3.maps)
        self.covlay5.calc_maps(self.poolinglay4.maps)
        self.fclay6.calc_maps(self.covlay5.maps)
        #self.outputlay7.rbf(self.fclay6.maps, mapclass)
        self.outputlay7.rbf_softmax(self.fclay6.maps)

    # Backward Propagation
    # mapclass : correct answer of mapset
    def bw_prop(self, mapset, mapclass, learn_rate) :
        # Set correct one value to 1 and other to 0
        output_error = zeros([1, 1, 10])
        output_error[0][0][mapclass] = 1
        
        #fclayer_error = self.outputlay7.back_propa(self.fclay6.maps, output_error, learn_rate, True)
        fclayer_error = self.outputlay7.back_propa_softmax(self.fclay6.maps, output_error, learn_rate, True)
        cov5_error = self.fclay6.back_propa(self.covlay5.maps, fclayer_error, learn_rate, True)
        pool4_error = self.covlay5.back_propa(self.poolinglay4.maps, cov5_error, learn_rate, True)
        cov3_error = self.poolinglay4.back_propa(self.covlay3.maps, pool4_error, learn_rate, True)
        pool2_error = self.covlay3.back_propa(self.poolinglay2.maps, cov3_error, learn_rate, True)
        cov1_error = self.poolinglay2.back_propa(self.covlay1.maps, pool2_error, learn_rate, True)
        ilayer_error = self.covlay1.back_propa(mapset, cov1_error, learn_rate, True)
        
    def print_netweight(self, filepath) :
        outputfile = open(filepath, 'w')
        cut_line = '\n-----------------------------------------------\n'
        outputfile.write(str(self.covlay1.covcores) + str(self.covlay1.covbias) + cut_line)
        outputfile.write(str(self.poolinglay2.poolparas) + cut_line)
        outputfile.write(str(self.covlay3.covcores) + str(self.covlay3.covbias) + cut_line)
        outputfile.write(str(self.poolinglay4.poolparas) + cut_line)
        outputfile.write(str(self.covlay5.covcores) + str(self.covlay5.covbias) + cut_line)
        outputfile.write(str(self.fclay6.weight) + str(self.fclay6.bias) + cut_line)
        outputfile.write(str(self.outputlay7.weight) + cut_line)

    def print_neterror(self, filepath) :
        outputfile = open(filepath, 'w')
        cut_line = '\n-----------------------------------------------\n'
        outputfile.write(str(self.covlay1.current_error) + cut_line)
        outputfile.write(str(self.poolinglay2.current_error) + cut_line)
        outputfile.write(str(self.covlay3.current_error) + cut_line)
        outputfile.write(str(self.poolinglay4.current_error) + cut_line)
        outputfile.write(str(self.covlay5.current_error) + cut_line)
        outputfile.write(str(self.fclay6.current_error) + cut_line)
        outputfile.write(str(self.outputlay7.current_error) + cut_line)
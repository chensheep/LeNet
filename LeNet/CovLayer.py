from numpy import *
from Layer import *


class CovLayer(Layer) :
        def __init__(self, lay_size = [], cov_core_sizes = [], mapcombindex = []) :
                Layer.__init__(self, lay_size) #Initialize maps, shape (6, 28, 28)

                self.covcores = []
                self.covbias = []
                self.mapcombindex = mapcombindex

                # Initialize the parameters of each convolution core & bias. 
                # The number -2.4/Fi and 2.4/Fi comes from the paper in Appendices A. See Fig.3.
                for cov_core_size in cov_core_sizes :
                        # Fi is from the definition in paper
                        Fi = cov_core_size[0] * cov_core_size[1] + 1
                        # Make random filters
                        self.covcores.append(random.uniform(-2.4/Fi, 2.4/Fi, cov_core_size)) 
                        # Make random biases
                        self.covbias.append(random.uniform(-2.4/Fi, 2.4/Fi))

                self.covcores = array(self.covcores)

                # covcores shape in Conv1 is (6, 1, 5, 5)


        # pre_maps is from previous layer
        # if current layer is first layer, pre_map is picture with 32 * 32
        def cov_op(self, pre_maps, covcore_index) :

                pre_map_shape = pre_maps.shape
                # Example :
                # Conv1 pre_maps shape is (1, 32, 32)
                # Conv3 pre_maps shape is (3, 14, 14) or (4, 14, 14) or (6, 14, 14)
                # Conv5 pre_maps shape is (16, 5, 5)

                
                covcore_shape = self.covcores[covcore_index].shape
                # Example :
                # Conv1 covcore_shape is (1, 5, 5)->6
                # Conv3 covcore_shape is (3, 5, 5)->6 (4, 5, 5)->9 (6, 5, 5)->1
                # Conv5 covcore_shape is (16, 5, 5)->120

                map_shape = self.maps[covcore_index].shape # [28, 28]
                # Example :
                # Conv1 map_shape is (28, 28)->6
                # Conv3 map_shape is (10, 10)->16
                # Conv5 map_shape is (1, 1)->120

                # Check the input size
                # If the result from input size match the output
                # Calculate the Convolution Layer
                # Example : 
                # Conv1 28 = 32 - 5 + 1
                # Conv3 10 = 14 - 5 + 1
                # Conv5 1 = 5 - 5 + 1
                if not (map_shape[-2] == pre_map_shape[-2] - covcore_shape[-2] + 1 \
                    and map_shape[-1] == pre_map_shape[-1] - covcore_shape[-1] + 1) :
                    return None

                # Scan through all pixels on per (feature) map , do convolution
                for i in range(map_shape[-2]) :
                        for j in range(map_shape[-1]) :

                                # Filter caculation in HW4 
                                localrecept = pre_maps[ : , i : i + covcore_shape[-2], j : j + covcore_shape[-1]]
                                # Example :
                                # pre_maps[ : , i : i + covcore_shape[-2], j : j + covcore_shape[-1]]
                                #           |
                                #           v
                                #       all channel
                                #
                                # i : i + covcore_shape[-2] -> 0 : 5 ... 27 : 32 (In Conv1)
                                #
                                # localrecept.shape in Conv1
                                # (1, 5, 5)
                                # localrecept.shape in Conv3
                                # (3, 5, 5) or (4, 5, 5) or (6, 5, 5)
                                # localrecept.shape in Conv5
                                # (16, 5, 5)

                                val = sum(localrecept * self.covcores[covcore_index]) + self.covbias[covcore_index]
                                # localrecept * self.covcores[covcore_index] -> element-wise product
                                # ref : https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html

                                # Use tanh(x) as activation function.
                                # Remark that tanh(x) = ((e^2x) -1)/((e^2x)+1)
                                # We use the parameters in the paper
                                # f(a) = Atanh(Sa), where A = 1.7159, S = 2/3, from the paper in Appendices A. See Fig.4.

                                val = exp((4.0/3)*val) 
                                self.maps[covcore_index][i][j] = 1.7159 * (val - 1) / (val + 1) 


        # The mapcombflag parameter is for deciding which method should use.
        # It's because the Conv Layers will use different way to filter the input. (Conv3 is special)
        def calc_maps(self, pre_mapset, mapcombflag = False) :

                #Example : 
                # self.maps.shape in Conv1 is (6, 28, 28)
                # self.maps.shape in Conv3 is (16, 10, 10)
                # self.maps.shape in Conv4 is (120, 1, 1)
                # len(self.maps) return shape[0] -> numbers of feature maps
                # pre_mapset.shap in Conv1 is (1, 32, 32) 
                # pre_mapset.shap in Conv3 is (6, 14, 14) 
                # pre_mapset.shap in Conv5 is (16, 5, 5)

                # mapcombflag = False, the first Conv Layer (Conv1, Conv5)
                if not mapcombflag :
                    for i in range(len(self.maps)) : # 輸出幾個feature做幾次卷積核
                        self.cov_op(pre_mapset, i)
                
                # Mapcombflag = True, the second Conv Layer (Conv3)
                else :
                    for i in range(len(self.maps)) :
                        self.cov_op(pre_mapset[self.mapcombindex[i]], i)
                        # Example : 
                        # pre_mapset[self.mapcombindex[i]].shape (3, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (3, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (3, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (3, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (3, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (3, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (4, 14, 14)
                        # pre_mapset[self.mapcombindex[i]].shape (6, 14, 14)


        # current_error shape must be same to shape of self.maps
        # each value in matrix current_error refers to error value of each neron in current layer 
        # which passed from the later layer, see Fig.5
        def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
                self.current_error = current_error

                # Flatten the array into one dim
                selfmap_line = self.maps.reshape([self.maps.shape[0] * self.maps.shape[1] * self.maps.shape[2]]) # channel * width * heigh
                currenterror_line = current_error.reshape([current_error.shape[0] * current_error.shape[1] * current_error.shape[2]])
                
                # Backward : df/da = 2/3 * 1.7159 * (1-(tanh(2/3*a)^2))
                # Where tanh(2/3*a) = selfmap_line[i]/1.7159 because self.maps[covcore_index][i][j] = 1.7159 * (val - 1) / (val + 1) 
                pcurrent_error = array([((2.0/3)*(1.7159 - (1/1.7159) * selfmap_line[i]**2))*currenterror_line[i]\
                                for i in range(len(selfmap_line))]).reshape(self.maps.shape)
                
                # Reset update data
                weight_update = self.covcores * 0
                bias_update = zeros([len(self.covbias)])
                # pre_error record error passed to previous layer
                pre_error = zeros(pre_mapset.shape)
                
                
                for i in range(self.maps.shape[0]) :
                        # If Conv2
                        if self.mapcombindex != [] :
                                pre_maps = pre_mapset[self.mapcombindex[i]]
                                select_pre_error = pre_error[self.mapcombindex[i]]
                        # If Conv1
                        else :
                                pre_maps = pre_mapset
                                select_pre_error = pre_error 
                        # Calculate Gradience.
                        for mi in range(self.maps.shape[1]) :
                                for mj in range(self.maps.shape[2]) :
                                    # get a core size from pre_maps
                                    cov_maps = pre_maps[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]
                                    # pcurrent_error * input = update value
                                    weight_update[i] += cov_maps * pcurrent_error[i][mi][mj]
                                    bias_update[i] += pcurrent_error[i][mi][mj]
                                    # caculate previous layer error pass to previous layer
                                    select_pre_error[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]\
                                                    += self.covcores[i] * pcurrent_error[i][mi][mj]
                # Update weights and biases
                if isweight_update :
                        self.covcores -= learn_rate * weight_update
                        self.covbias -= learn_rate * bias_update
                return pre_error
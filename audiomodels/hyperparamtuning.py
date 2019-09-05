from  model import *

#GRAPH BUILDER AND RUNNER CLASS

class Overlord:
    '''
     Overlord Class uses Builder class to build each architecture block of the
     network and build the final model that will be trained.
    '''
    def __init__(self, vv_vv, oOoOoO, P_b_P, **kwargs):
        self.slicer = vv_vv
        self.dataloader = oOoOoO
        self.hyperparameters = P_b_P
        self.dict = kwargs['777']

    def iron_man(self):
        '''
         Function that use given parameters to build final model to be trained
        '''
        pass

# Hyperparam choosing class
class Hyperparameter(object):
    '''
     Template class for hyperparam selection.

     Default Args:
         - frame_data: if data_type is image (or 'img'), frame_data is an
           recipe to frame the data input, each frame will have its own
           architectural building block sequence for the model.
           'width', 'height', 'both' or 'None'.
         - frame_selection: strategy to select frames generated from
           DataSlicer, given as keywordarg.

           'all' to select all frames.
           'random' to select random portion of the frames, if not given it
           takes 0.7 as default.
           'fraction' takes, 0.7 by default, a fraction of the frames, taking
           the bigger to the smaller in size.

           kwargs:
               - portion: if not given defaults to 0.7. The amount of frames
                 to be taken.
               - slicer: DataSlicer object o


    '''

    def __init__(self, data_type, frame_data = 'width',
            frame_selection='all', **kwargs):
        self.data_type = data_type
        self.frame_data = frame_data
        self.frame_selection = frame_selection
        self.hyperparameters = kwargs

    def __call__(self, directive):
        pass

    def configure(self):
        pass

    def frame_selector(self):
        pass

class GCNNMaxPooling(Hyperparameter):
    '''
      Hyperparameter Selection  for Gated Convolutional Neural Network with Max
     Pooling 2D.

      Use the caller with length 2 directive sequence (a name and a value, the name
      must be a string) to get hyperiparameters tunners.

      The Directive are

        if directive[0] == 'num_bank':
            return self.filterbank[directive[1]] # List with filter bank sizes
        from one of the data dimensions

        if directive[0] == 'maximal_filter':
            return self.max_filter[directive[1]] #  Maximal value of filter
            bank to search for hyperparametrization in respeective bank block

        if directive[0] == 'minimal_filter':
            return self.min_filter[directive[1]] # Minimal value to search for
        filter bank,  defaults to 3.

        if directive[0] == 'maximal_bank_blocks': 
            return self.max_blocks # Maximal number of convolution blocks. If
            it does not suffice to transform to an embedding array them will
            to convolve the final data representation and output the embeded
            1-D array.

        if directive[0] == 'minimal_bank_blocks':
            return self.min_blocks #  Minimal number of convolution blocks
        
        if directive[0] == 'gate_positions':
            # 0 for Gate presence 1 for not present
            return self.gate_positions[directive[1]]

        if directive[0] == 'maximal_number_of_gates':
            # Equivalent to the sum of elements of gate_positions array
            return self.num_of_gates

    '''
    def __init__(self, data_type = 'img', frame_data = 'width', **kwargs):
        super( GCNNMaxPooling, self).__init__(data_type = data_type,
                frame_data= 'width', **kwargs)

    def __call__(self, directive):

        if directive[0] == 'num_bank':
            return self.filterbank[directive[1]]

        if directive[0] == 'maximal_filter': 
            return self.max_filter[directive[1]]

        if directive[0] == 'minimal_filter':
            return self.min_filter[directive[1]]

        if directive[0] == 'maximal_bank_blocks':
            return self.max_blocks

        if directive[0] == 'minimal_bank_blocks':
            return self.min_blocks

        if directive[0] == 'gate_positions':
            # 0 for Gate presence 1 for not present
            return self.gate_positions[directive[1]]

        if directive[0] == 'maximal_number_of_gates':
            # Equivalent to the sum of elements of gate_positions array
            return self.num_of_gates

        if directive[0] == 'frame':
            # Return frame size of the dimension width
            return self.frame

        if directive[0] == 'dim':
            return self.dim

    def estimate_memusage(self):
        '''
         Researching better way to estimate memory usage. Not implemented yet
        '''
        pass

    def configure(self):
        '''

        '''
        pass

    def setup_filterbanks(self):
        pass


from  model import *
import warnings
#GRAPH BUILDER AND RUNNER CLASS


# TODO- Refatorar funcoes de extracao de parametros
def _keyargs_extract(kwargs):

    kw = kwargs.keys()

    try:
        portion = kwargs[kw[kw.index('portion')]]
        instance = isinstance(portion,float)
        assert instance
        assert portion < 1.0 and portion > 0.2
    except:
        portion = 0.7

    try:
        slicer = kwargs[kw[kw.index('slicer')]]
        instance = isinstance(slicer, DataDomainSlicer)
        assert instance
        assert len(slicer.slices) > 0
    except:
        warnings.warn('Warning: DataDomainSlicer object not give, setting a\
        LadderSlicer by default. Reset the slicer bounds, number of steps,\
        and run slicer.configure()')
        slicer = LadderSlicer(0,0,0)

    return portion,slicer

class Model:
    '''
      Model Class uses Builder class to build each architecture block of the
     network and build the final model that will be trained.
    '''
    def __init__(self, **kwargs):
        #self.slicer = vv_vv
        #self.dataloader = oOoOoO
        #self.hyperparameters = P_b_P
        pass

# Hyperparam choosing class
class Hyperparameter(object):
    '''
     Template class for hyperparam selection.

     Default Args:
         - frame_data: Strategy to frame data, E.G.If data_type is image (or
           'img'), each frame will have its own
           architectural building block sequence for the model.
           'width', 'height', 'both' or 'None'.
           TODO: Implement this part for general case.

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
               - slicer: DataDomainSlicer object. The slices index of the data
                 frame domain, if not given defaults to LadderSlicer.
    '''

    def __init__(self, data_type, frame_data = 'width',
            frame_selection='all', **kwargs):
        self.data_type = data_type # Not in Use
        self.frame_data = frame_data # Not in Use
        self.frame_selection = frame_selection
        self.hyperparameters = kwargs
        self.configure()

    def __call__(self, directive):
        pass 

    def configure(self):
        hp = self.hyperparameters
        portion,slicer = _keyargs_extract(hp)
        if not (frame_data == 'None'):
            self.portion = portion
            self.slicer = slicer
            self.slicer.configure()
            assert len(self.slicer.slices) > 0,'DataDomainSlicer is not \
            configured. Reset slicer with the bounds and number of steps \
            ,configure it and run frame_selector'
            self.frame_selector()

    def frame_selector(self):

        if frame_selection == 'fraction':
            number_of_frames = int(self.portion*len(self.slicer.slices))
            size = lambda  x: [el[1]-el[0] for el in x]
            slice_sizes = size(self.slicer.slices)
            ordered_sizes_index = np.argsort(slice_sizes)
            select = []
            for i in range(number_of_frames):
                select.append(self.slicer.slices[ordered_sizes_index[len(slice_sizes)-i-1]])
            self.slicer.slices = select 

        if frame_selection == 'random':
            number_of_frames = int(self.portion*len(self.slicer.slices))
            select = []
            for i in range(number_of_frames):
                select.append( self.slicer.slices.pop( np.random.randint(
                    len(self.slicer.slices) ) ) )
            self.slicer.slices = select

class GCNNMaxPooling(Hyperparameter):
    '''
      Hyperparameter Selection  for Gated Convolutional Neural Network with Max
     Pooling 2D.

    '''
    def __init__(self, data_type = 'img', frame_data = 'width', **kwargs):
        super( GCNNMaxPooling, self).__init__(data_type = data_type,
                frame_data= frame_data, **kwargs)
        self.configure()

    def __call__(self, directive):
        pass

    def estimate_memusage(self):
        '''
         Researching better way to estimate memory usage. Not implemented yet
        '''
        pass

    def configure(self):
        '''
        Configure by Default with a LadderSlicer, the maximal and minimal
        bounds of the Data Domain must be given in kwargs
        '''
        super(GCNNMaxPooling, self).configure()
        self.extract_hparams()

    def extract_hparams(self):
        '''
        Extract the given hyperparameters for the GCNN-MaxPooling Architecture.
        Includes:

            - conv2D_blocks: sequence of sequences, each inner sequence has
              elements of size 2 with the filter size for each dimension.
            - gated_positions: sequence of the size of  conv2D_blocks with
              elements 1, for presence of gating, 0 for absence.
        '''

        kw = self.hyperparameters.keys()
        ### Convolutions and Max Poolings Initial/Given Condition
        try:
            self.conv2D_blocks = self.hyperparameters[kw[kw.index('conv2D_blocks')]]
            sequence = hasattr(self.conv2D_blocks, '__iter__')
            assert sequence, 'conv2D_blocks is not sequence container'
            for block in self.conv2D_blocks:

                sequence = hasattr(block, '__iter__')
                assert sequence, 'element of conv2D_blocks must be a sequence of\
                    filter size elements.'

                for filter_size in block:
                    sequence = hasattr(filter_size,'__iter__')
                    assert sequence
                    assert len(filter_size) == 2,'filter_size must be a sequence with size 2.'

        except:
            self.conv2D_blocks = [[(3,3),(3,3)], [(2,2)]]

        self.pooling = []
        for i in len(self.conv2D_blocks-1):
            self.pooling.append(2)

        ####


        ### Gate position vector.
        try:
            # list of 1s and 0s, with 1s in the index of block where the gate
            # is present
            gated = self.hyperparameters[kw[kw.index('gated_positions')]]
            instance = hasattr(gated, '__iter__')
            assert instance
            self.gated_positions = gated
            self.num_of_gates = sum(gated)
        except:
            warnings.warn('Warning: Gate positions not given, Gate position \
                must provided as a sequence of 1s and 0s of the size of the \
                number of filterbanks, 0 for absence of gate, 1 for presence.\
                Use set_gating(gatelist) for this.')
            iself.gated_positions = len(self.conv2D_blocks)*[0]
            self.num_of_gates = sum(self.gated_positions)

        ####

    def set_gating(self, gatelist):

        assert hasattr(gatelist, '__iter__'),'gated_position must be a sequence'
        assert (isinstance(gatelist[0], int) or isinstance(gatelist[0],
            float)),'gated_position elements are float or int: either 1, for there is gate, 0\
        for no gate, any other value is transformed for 1, in case of being >\
        1, or 0 in case of being < 0.'

        self.gated_positions = gatelist
        p = np.array(self.gated_positions)

        # test for gated_positions having  numbers others than 1 and 0,
        # turn elements greater than 1 to 1 and less than 0 to 0
        if ((p < 0).any() or (p>1).any()):
            p[p>1] = 1
            p[p<0] = 0
            self.gated_positions = list(p)
            del p

        return self.gated_positions

    def update_architecture(self, frame):
        '''
         Use slicer.slices to build  self.conv2D_blocks and add it to
         Model configuration.
        '''
        pass


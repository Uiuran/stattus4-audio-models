from  model import *
import warnings
#GRAPH BUILDER AND RUNNER CLASS


# TODO- Refatorar funcoes de extracao de parametros
# TODO- fazer META tags: config frames, blocos configs e gates configs para
# casos de implementacao do Hyperparameter tuning
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

    def __init__(self, data_type, domain_compact, frame_data = 'width',
            frame_selection='all', **kwargs):
        self.data_type = data_type # Not in Use
        self.frame_data = frame_data # Not in Use
        self.frame_selection = frame_selection
        self.hyperparameters = kwargs
        self.configure()

    def eval_metrics(self):
        pass

    def configure(self):
        hp = self.hyperparameters
        portion,slicer = _keyargs_extract(hp)
        if not (self.frame_data == 'None'):
            self.portion = portion
            self.slicer = slicer
            self.slicer.configure()
            assert len(self.slicer.slices) > 0,'DataDomainSlicer is not \
            configured. Reset slicer with the bounds and number of steps \
            ,configure it and run frame_selector'
            self.frame_selector()

    def search(self, frame, algorithm):
        '''
         Algorithm for Neural Architecture Search function. To be implemented
         according to the HyperParamTuning strategy
        '''
        pass

    def eval_frame(self, frame, algorithm):
        '''
          Function to be implemented by heirs of this function.
         eval_frame evaluates the frame and update its architecture according to given algorithm.
         update_architecture retrieve a frame and use eval_frame to do the
         update.
        '''
        pass

    def eval_metrics(self, frame, config, mode):
        '''
          Evaluate metrics for a given configuration of architecture, for given
         frame.
          mode argument must be implemented as a selector of which metrics, on
          the given architecture configuration, will be used, since it can be
          many helpful metrics to automatically search for a architecture
          configuration.
          E.G. one can need eval only the last block of filter
          banks to know if you need to grow this block (according to a
          algorithm), or you can eval all the configurational blocks to
          conclude if the configuration is deep enough for that frame.
        '''
        pass

    def update_architecture(self, frame, algorithm):
        '''
         Retrieve frame and use eval_frame to search for architecture according
         to the algorithm implemented in the search function.

         eval_frame, and eval_metrics must also be implemented according to the
         architecture in which you will do hyperparametrization searching.
        '''
        assert len(self.slicer.slices)>0,'Data framing is not properly setted\
        o self.slicer.reset(max,min,number_of_steps) them try updating again'

        if isinstance(frame,type(None)):
            try:
                frame = self.slicer.get_slice()
            except:
                # Inu generator exception re-setup slicer, with given frame selection
                # by default. This directive will only change in case of necessity
                self.slicer.configure()
                self.frame_selector()
                frame = self.slicer.get_slice()

            self.eval_frame(frame, algorithm)

        elif hasattr(frame,'__iter__'):

            assert len(frame) == 2,'not a sequence of length 2 corresponding to\
                a frame'

            if (isinstance(frame[0],int) or isinstance(frame[0],float)) and\
                (isinstance(frame[1],int) or isinstance(frame[1],float)) and\
                (frame[0] < frame[1]):

                self.eval_frame(frame, algorithm)

            else:
                raise('Not a numbered frame. The two element of the sequence\
                      must be integers of type int or float and element 1 must\
                      be bigger than element 0.')


    def frame_selector(self):

        if self.frame_selection == 'fraction':
            number_of_frames = int(self.portion*len(self.slicer.slices))
            size = lambda  x: [el[1]-el[0] for el in x]
            slice_sizes = size(self.slicer.slices)
            ordered_sizes_index = np.argsort(slice_sizes)
            select = []
            for i in range(number_of_frames):
                select.append(self.slicer.slices[ordered_sizes_index[len(slice_sizes)-i-1]])
            self.slicer.slices = select

        if self.frame_selection == 'random':
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
    def __init__(self, data_type='img', domain_compact=[(0.0,200.0),(0.0,200.0)], frame_data = 'width',
                 frame_selection='fraction', **kwargs):
        super( GCNNMaxPooling, self).__init__(data_type=data_type,
                                              domain_compact=domain_compact,
                                              frame_data= frame_data,
                                              frame_selection=frame_selection, **kwargs)
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

            - conv2D_blocks: dictionary with a frame as key and sequence of
              sequences has value, each inner sequence elements of size 2 with
              the filter size for each dimension.

            - gated_positions: sequence of the size of  conv2D_blocks with
              elements 1, for presence of gating, 0 for absence.
        '''

        kw = self.hyperparameters.keys()
        ### Convolutions and Max Poolings Initial/Given Condition
        try:
            self.conv2D_blocks = self.hyperparameters[kw[kw.index('conv2D_blocks')]]
            isdict = hasattr(self.conv2D_blocks,'values')
            assert isdict, 'conv2D_blocks is not dictionary'
            sequence = hasattr(self.conv2D_blocks.values(), '__iter__')
            assert sequence, 'conv2D_blocks is not sequence container'
            for block in self.conv2D_blocks.values():

                sequence = hasattr(block, '__iter__')
                assert sequence, 'element of conv2D_blocks must be a sequence of\
                    filter size elements.'

                for filter_size in block:
                    sequence = hasattr(filter_size,'__iter__')
                    assert sequence
                    assert len(filter_size) == 2,'filter_size must be a sequence with size 2.'

        except:
            self.conv2D_blocks = {}#[[(3,3),(3,3)], [(2,2)]]

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
            self.gated_positions = len(self.conv2D_blocks.values())*[0]
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

    def search(self, frame, algorithm):

        self.eval_metrics(frame,self.conv2D_blocks[frame], mode ='all')

        while self.degrad < 0.9:

            self._search_eval_last(frame)
            self.eval_metrics(frame,self.conv2D_blocks[frame], mode ='all')
            if self.degrad > 0.9:
                break
            self._search_eval_all(frame)
            self.eval_metrics(frame,self.conv2D_blocks[frame], mode ='all')

    def _search_eval_last(self, frame):
        deep = 0
        filtersize = 1
        c = 0
        self.degrad = 0.0
        while self.degrad < 0.1:

            self.eval_metrics(frame,self.conv2D_blocks[frame], mode ='last')

            if filtersize:
                for i in range(len(self.conv2D_blocks[frame][-1])):
                    self.conv2D_blocks[frame][-1][i] = list(self.conv2D_blocks[frame][-1][i])
                    self.conv2D_blocks[frame][-1][i][1] = self.conv2D_blocks[frame][-1][i][1] + 2
                    self.conv2D_blocks[frame][-1][i] = tuple(self.conv2D_blocks[frame][-1][i])

                # Interchanging strategies of deepening and filtersize growing
                # for each block
                filtersize = 0
                deep = 1

            if deep:
                self.conv2D_blocks[frame][-1].append(self.conv2D_blocks[frame][-1][0])
                deep = 0
                filtersize = 1

    def _search_eval_all(self, frame):

            self.degrad = 0.0
            l = len(self.conv2D_blocks[frame][-1])
            self.conv2D_blocks[frame].append([])

            for i in range(l):
                self.conv2D_blocks[frame][-1].append( (\
                    2,\
                    2) )


    def eval_frame(self, frame, algorithm):
        '''
          If network has this frame as input, evaluate it's metrics on respective
         architecture and tries to expand it.
        '''

        self.degrad = 0.0
        if self.conv2D_blocks.has_key(frame):

            # Adaptative blocks, first try on piramidal logic, further studies
            # on Architecture search
            self.search(frame, algorithm)

        else:
            self.conv2D_blocks[frame] = [[(3,3)]]
            self.search(frame, algorithm)

    def eval_metrics(self, frame, config, mode = 'all'):
        '''
         GCNN with Maxpooling metrics for configuring an architecture block.
         This implementation is assuming framing on 'width' only.
        '''
        #TODO- eval metrics for other frame dimensions

        b = frame[1]-frame[0]
        a=b
        c = 0

        if mode == 'all':
            for block in config:
                degrad = calculatefilter(a, block, dim=1)
                self.degrad = ( b - (degrad) )/float( b )
                # exact integer division by 2 due to MaxPooling
                a = degrad
                a = a//2
                c += 1

                if a < 0:
                    print(self.degrad)
                    print('negative output dimension for block number ',c)
                    print(block)
                    self.last_block = c-1 
                    break

            self.degrad = ( b - (degrad) )/float( b )

        if mode == 'last':
            block = config[-1]
            degrad = calculatefilter(a, block, dim=1)
            self.degrad = ( b - (degrad) )/float( b )
            # exact integer division by 2 due to MaxPooling

            if degrad < 0:
                print('negative output dimension for block')
                print(block)

    def update_architecture(self, frame=None, algorithm = 'pyramid'):
        '''
         Retrieve frame and use eval_frame to search for architecture according
         to the algorithm implemented in the search function.

         eval_frame, and eval_metrics must also be implemented according to the
         architecture in which you will do hyperparametrization searching.
        '''
        super( GCNNMaxPooling, self).update_architecture(frame=frame,algorithm=algorithm)

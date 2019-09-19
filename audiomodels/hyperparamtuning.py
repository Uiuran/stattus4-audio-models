from  model import *
from util import *
import warnings
import copy
#GRAPH BUILDER AND RUNNER CLASS


# TODO- fazer META tags: config frames, blocos configs e gates configs para
# casos de implementacao do Hyperparameter tuning

# Hyperparam choosing class
class Hyperparameter(object):
    '''
     Template class for hyperparam selection.

     Default Args:

         - domain_compact: a sequence of sequences, each sequence correspond to
         a dimension of the data structure to be processed by the
         architecture, they must have length 2 specifing lower bound to upper
         bound of the sequence.

     kwargs:

         - slicer: DataDomainSlicer object. The slices index of the data
           frame domain, if not given defaults to LadderSlicer.

         - frame_selection: strategy to select frames generated from
         DataSlicer, given as keywordarg.

           'all' to select all frames.

           'random' to select random portion of the frames, if not given it
           takes 0.7 as default.

           'fraction' takes, 0.7 by default, a fraction of the frames, taking
           the bigger to the smaller in size.

    '''
    # Configurar slicer, ver como passar
    def __init__(self, domain_compact, **kwargs):
        self.data_domain = domain_compact
        self.hyperparameters = kwargs
        self.deep_arch = {}

    def assert_frame(self):
        pass

    def configure(self):
        hp = self.hyperparameters
        # Try to find domain slicer from keyword args
        # if not found then return the entire domain as a unique frame
        self._configure_slicer(hp)
        #assert len(self.slicer.slices) > 0,'DataDomainSlicer is not \
           # configured. Reset slicer with the bounds and number of steps \
           # ,configure it and run frame_selector'
        self.slicer.frame_selector()

    def search(self, frame, algorithm):
        '''
         Algorithm for Neural Architecture Search function. To be implemented
         according to the HyperParamTuning strategy
        '''
        pass

    @staticmethod
    def _std_in(*args,**kwargs):
        '''
         Standart input for Architecture.
        '''
        pass

    def _configure_slicer(self, hyperparameter):

        if hyperparameter.has_key('recursive_depth'):
            self.recursive_depth = hyperparameter['recursive_depth']
        else:
            # Default is 2, passed in the class implementation, however you may
            # implement it here
            self.recursive_depth = 2

        if hyperparameter.has_key('frame_selection'):
            self.frame_selection = hyperparameter['frame_selection']
        else:
            # The default value for frame_selection is implemented in Slicer
            # classes however one can optionally pass it here for
            # hyperparametrization
            self.frame_selection = 'fraction'

        if hyperparameter.has_key('frame_fraction'):
            self.frame_fraction = hyperparameter['frame_fraction']
        else:
            # Same consideration as above, about default value for fraction of
            # frames applies here
            self.frame_fraction = 0.15

        if hyperparameter.has_key('slicer'):
            self.slicer = hyperparameter['slicer']
            # Try to find slicer args

            if isinstance(self.slicer,type) and self.slicer.__name__ ==\
                'EmbeddedSlicer':
                if hyperparameter.has_key('number_of_steps'):
                    assert isinstance(hyperparameter['number_of_steps'],int),'number of steps must be integer'
                    number_of_steps = hyperparameter['number_of_steps']
                else:
                    number_of_steps = 10

                if hyperparameter.has_key('fater_slicer') and hyperparameter.has_key('mater_slicer'):
                    assert isinstance(hyperparameter['fater_slicer'],type) and\
                        isinstance(hyperparameter['mater_slicer'],type),'fater and mater slicers must be an instance of type'
                    mater_slicer = hyperparameter['mater_slicer']
                    fater_slicer = hyperparameter['fater_slicer']
                else:
                    mater_slicer = LadderSlicer
                    fater_slicer = LadderSlicer

                self.slicer = self.slicer(self.data_domain,number_of_steps,
                                          mater_slicer,fater_slicer,
                                          frame_selection=self.frame_selection,frame_fraction=self.frame_fraction,
                                          recursive_depth=self.recursive_depth)
            elif isinstance(self.slicer,type) and not self.slicer.__name__ ==\
                'EmbeddedSlicer':
                if hyperparameter.has_key('number_of_steps'):
                    assert isinstance(hyperparameter['number_of_steps'],int),'number\
                        o f steps must be integer'
                    number_of_steps = hyperparameter['number_of_steps']
                else:
                    number_of_steps = 10
                self.slicer = self.slicer(self.data_domain, number_of_steps,
                                          frame_selection=self.frame_selection,frame_fraction=self.frame_fraction)
        else:
            self.slicer = NoSliceSlicer(self.data_domain)

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

    def eval_frame(self, frame, algorithm):
        '''
         Helper function to retrieve frame and implement search algorithm.
        '''
        self.assert_frame()
        # HASHABLE TYPE to KEY DICT OF BLOCKS
        frame = tuple( [tuple(el) for el in frame] )
        if self.deep_arch.has_key(frame):

            # Adaptative blocks, first try on piramidal logic, further studies
            # on Architecture search
            self.search(frame, algorithm)

        else:
            self.deep_arch[frame] = self._std_in()
            self.search(frame, algorithm )

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
                self.slicer.configure(self.data_domain)
                self.slicer.frame_selector()
                frame = self.slicer.get_slice()

            self.eval_frame(frame, algorithm)

        elif hasattr(frame,'__iter__'):

            for i in range(len(frame)):
                assert len(frame[i]) == 2,'not a sequence of length 2 corresponding to a frame'

                assert (isinstance(frame[i][0],int) or isinstance(frame[i][0],float)) and\
                    (isinstance(frame[i][1],int) or isinstance(frame[i][1],float)) and\
                    (frame[i][0] < frame[i][1]), 'frame dimension bounds must\
                be numeric type and upper bound must be bigger than lower bound'

            self.eval_frame(frame, algorithm)

class GCNNMaxPooling(Hyperparameter):
    '''
      Hyperparameter Selection  for Gated Convolutional Neural Network with Max
     Pooling 2D.
    '''
    def __init__(self, domain_compact, **kwargs):
        super( GCNNMaxPooling, self).__init__(domain_compact, **kwargs)
        self.deep_arch['arch_type'] = 'GCNNMaxPooling'
        self.deep_arch['deep_type'] = ['convolution','gate']
        self.configure()

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

    def assert_frame(self):
        self.assert_is2Dframe()

    def assert_is2Dframe(self):
        assert self.slicer.dim == 2,'Frame is of dimensionality\
            {}'.format(self.slicer.dim)+', must have 2 dimensions'

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
            self.deep_arch = self.hyperparameters[kw[kw.index('conv2D_blocks')]]
            isdict = hasattr(self.deep_arch,'values')
            assert isdict, 'conv2D_blocks is not dictionary'
            sequence = hasattr(self.deep_arch.values(), '__iter__')
            assert sequence, 'conv2D_blocks is not sequence container'
            for block in self.deep_arch.values():

                sequence = hasattr(block, '__iter__')
                assert sequence, 'element of conv2D_blocks must be a sequence of\
                    filter size elements.'

                for filter_size in block:
                    sequence = hasattr(filter_size,'__iter__')
                    assert sequence
                    assert len(filter_size) == 2,'filter_size must be a sequence with size 2.'

        except:
            self.deep_arch = {}#[[(3,3),(3,3)], [(2,2)]]

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
            self.gated_positions = len(self.deep_arch.values())*[0]
            self.num_of_gates = sum(self.gated_positions)

        ### Max Pooling Cut-off

        if self.hyperparameters.has_key('pooling_cutoff'):
            self.pooling_cutoff = self.hyperparameters['pooling_cutoff']
        else:
            self.pooling_cutoff = 25

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

    def _error_fix(self, error, frame, blockindex):
        '''
         Helper function to correct automatic architecture search.
        '''
        l = len(self.deep_arch[frame][blockindex])
        for j in range(self.slicer.dim):
            c = 0
            while  error[j] != -1:
                self.deep_arch[frame][blockindex][error[j]] = list(self.deep_arch[frame][blockindex][error[j]])
                self.deep_arch[frame][blockindex][error[j]][j] = self.deep_arch[frame][blockindex][error[j]][j] - 1
                self.deep_arch[frame][blockindex][error[j]] = tuple(self.deep_arch[frame][blockindex][error[j]])
                if c == 0 and error[j]+1 < l:
                    for i in range(error[j]+1,l):
                        self.deep_arch[frame][blockindex][i] = list(self.deep_arch[frame][blockindex][i])
                        self.deep_arch[frame][blockindex][i][j] = 1
                        self.deep_arch[frame][blockindex][i] = tuple(self.deep_arch[frame][blockindex][i])
                    c = 1
                error = self.eval_metrics(frame, self.deep_arch[frame], mode ='all')

    def search(self, frame, algorithm):
        '''
         Function that implements algorithm search for architecture by
         evaluating frames and metrics
        '''
        self.assert_frame()
        error = self.eval_metrics(frame,self.deep_arch[frame], mode ='all')
        self._error_fix(error, frame, -1)

        arr = lambda x: np.array([0.1 if ((x[i][1]-x[i][0])//10.0 or (x[i][1]-x[i][0])//10.0) > 0 else 1.0/(10*(x[i][1]-x[i][0])) for i in range(len(x))])
        while (np.array(self.degrad) < 8.*arr(frame)).any():
            self._search_eval_last(frame)
            error = self.eval_metrics(frame,self.deep_arch[frame], mode ='all')
            self._error_fix(error, frame, -1)

            if (np.array(self.degrad) > 8.*arr(frame)).all():
                break

            self._search_append_block(frame)
            error = self.eval_metrics(frame,self.deep_arch[frame], mode ='all')
            self._error_fix(error, frame, -1)

    def _search_eval_last(self, frame):
        deep = 0
        filtersize = 1
        c = 0
        self.degrad = [0.0]*self.slicer.dim

        arr = lambda x: np.array([0.1 if ((x[i][1]-x[i][0])//10.0 or (x[i][1]-x[i][0])//10.0) > 0 else 1.0/(10*(x[i][1]-x[i][0])) for i in range(len(x))])
        while (np.array(self.degrad) < 2.*arr(frame)).all():
            if filtersize:
                for i in range(len(self.deep_arch[frame][-1])):
                    self.deep_arch[frame][-1][i] = list(self.deep_arch[frame][-1][i])
                    for j in range(self.slicer.dim):
                        self.deep_arch[frame][-1][i][j] = self.deep_arch[frame][-1][i][j] + 1
                    self.deep_arch[frame][-1][i] = tuple(self.deep_arch[frame][-1][i])

                error = self.eval_metrics(frame,self.deep_arch[frame], mode ='last')
                if (np.array(self.degrad) > 2.*arr(frame)).all():
                    break

                filtersize = 0
                deep = 1

            if deep:
                self.deep_arch[frame][-1].append(self.deep_arch[frame][-1][0])
                error = self.eval_metrics(frame,self.deep_arch[frame], mode ='last')

                if (np.array(self.degrad) > 2.*arr(frame)).all():
                    break
                deep = 0
                filtersize = 1

    def _search_append_block(self, frame):

        self.degrad = [0.0]*self.slicer.dim
        self.deep_arch[frame].append([])
        self.deep_arch[frame][-1].append((2,2))

    def eval_frame(self, frame, algorithm):
        '''
        Helper function to retrieve a frame and apply some algorithm of
        architecture search.
        '''
        super(GCNNMaxPooling, self).eval_frame(frame, algorithm)

    @staticmethod
    def _std_in(*args,**kwargs):
        '''
         Standart input for Architecture.
        '''
        return [[(2,2)]]



    def _degradation_metrics(self, frame, config, mode):

        sizes = [frame[i][1]-frame[i][0] for i in range(self.slicer.dim)]
        a = copy.copy(sizes)
        c = 0


        if mode == 'all':
            for block in config:
                degrad,error = calculatefilter(a, block)
                self.degrad = [( sizes[s] - (degrad[s]) )/float( sizes[s] ) for s in range(len(sizes))]
                # exact integer division by 2 due to MaxPooling
                a = degrad

                for i in range(len(a)):
                    if not (error[i] == -1):
                        c=1
                    if int(a[i]) > self.pooling_cutoff:
                        a[i] = a[i]//2
                if c == 1:
                    break

            return error

        if mode == 'last':

            block = config[-1]
            degrad,error = calculatefilter(a, block)
            self.degrad = [( sizes[s] - (degrad[s]) )/float( sizes[s] ) for s in range(len(sizes))]
            return error

    def eval_metrics(self, frame, config, mode):
        '''
         GCNN with Maxpooling metrics for configuring an architecture block.

         mode:
             'all': run degradation metrics against the full configuration
             given.
             'last': run degradation metrics against the last block of
             configuration.
        '''

        self.assert_frame()
        error = self._degradation_metrics(frame, config, mode)
        return error

    def update_architecture(self, frame=None, algorithm = 'pyramid'):
        '''
         Retrieve frame and use eval_frame to search for architecture according
         to the algorithm implemented in the search function.

         eval_frame, and eval_metrics must also be implemented according to the
         architecture in which you will do hyperparametrization searching.
        '''
        self.assert_frame()
        super( GCNNMaxPooling, self).update_architecture(frame=frame,algorithm=algorithm)

from template import NotDomainSlicerError, WrongSlicerError
import warnings
import numpy  as np

class DataDomainSlicer(object):

    def __init__(self, data_domain, number_of_steps, frame_selection, frame_fraction):
        '''
         Abstract Class.
         Get domain bounds and set the domain slicing method.
        '''
        self.frame_selection = frame_selection
        self.frame_fraction = frame_fraction
        self.number_of_steps = number_of_steps

    def assert_domain(self, data_domain):
        '''
          Assert that given data_domain is a sequence of sequences, each having
         length 2 and type int or float for lower bound and upper bound of the
         compact dimensions.
        '''

        assert hasattr(data_domain,'__iter__'),'data_domain must be a\
            sequence of with the length of data dimensionality.'
        self.dim = len(data_domain)

        for i in range(self.dim):
            assert hasattr(data_domain[i],'__iter__') and\
                len(data_domain[i])==2,'dimension {}'.format(i)+' bounds\
                must be a sequence of length 2'
            assert isinstance(data_domain[i][0], int) or\
                isinstance(data_domain[i][0],float),'dimension {}'.format(i)+' bounds\
                must be numerical value'
            assert data_domain[i][0] < data_domain[i][1],'dimension {}'.format(i)+' bounds\
               must be ordered, that is the element 0 must be smaller them the\
               1'

        for domain_bound in data_domain:
            self.max.append( domain_bound[1] )
            self.min.append( domain_bound[0] )

    def configure(self, data_domain):
        '''
         Configure the slicing based on the mode and the dimension supreme and
        infimum. In practice it returns a list of tuples, to attribute slices, each with the lower
        and the upper bound of the slice.

        abstract method to be implemented
        '''
        self.slices = []
        self.max = []
        self.min = []
        self._n = 0
        self.assert_domain(data_domain)

    def set_number_of_steps(self, steps_num):
        self.number_of_steps = steps_num

    def reset_domain(self, data_domain):
        self.assert_domain(data_domain)

    def reset(self, data_domain, steps_num):
        '''
          Take new data_domain number of steps(frames) and output reconfigure
         the slicer.
        '''
        self.set_number_of_steps(steps_num)
        self.configure(data_domain)

    def get_slice(self):
        '''
        Returns tuple with 2 elements corresponding to the lower and
        upper bound of the slice respectively. The number of bounds and the
        size  of the slice depends upon slice mode.

        abstract method to be implemented
        '''
        if self._n < self.number_of_steps:
            slic = self.slices[self._n]
            self._n += 1
            return tuple(slic)
        else:
            raise StopIteration()

    def next(self):
        return self.get_slice()

    def __iter__(self):
        return self

    @staticmethod
    def frame_size(frame):
        size = 1.0
        for dim in frame:
            size = (dim[1]-dim[0])*size

        return size

    def frame_selector(self):

        if self.frame_selection == 'fraction':
            number_of_frames = int(self.frame_fraction*len(self.slices))

            size_list = lambda x: [self.frame_size(el) for el in x]
            slice_sizes = size_list(self.slices)
            ordered_sizes_index = np.argsort(slice_sizes)
            select = []
            for i in range(number_of_frames):
                select.append(self.slices[ordered_sizes_index[len(slice_sizes)-i-1]])

            self.slices = select


        if self.frame_selection == 'random':
            number_of_frames = int(self.frame_fraction*len(self.slices))
            select = []
            for i in range(number_of_frames):
                select.append( self.slices.pop( np.random.randint(
                        len(self.slices) ) ) )
            self.slices = select

class LadderSlicer(DataDomainSlicer):
    '''
      Slice the domain according to ladder-wise steps, that is, each step is
     half superpose with the previous step. Returns  steps according to the
     step_size and the given number of  steps
    '''

    def  __init__(
        self,
        data_domain,
        number_of_steps,
        frame_selection,
        frame_fraction
    ):
        super(LadderSlicer, self).__init__(
            data_domain,
            number_of_steps,
            frame_selection,
            frame_fraction
        )
        self.configure(data_domain)

    def configure(self, data_domain):
        '''
         Configure the slicing based on the mode and the dimension supreme and
        infimum. In practice it returns a list of tuples, to attribute slices, each with the lower
        and the upper bound of the slice.
        '''
        super( LadderSlicer, self).configure(data_domain)
        self.step_size = []

        for i in range(self.dim):
            self.step_size.append(int(2*(self.max[i] -\
                                         self.min[i])/(self.number_of_steps+1)))
            while self.step_size[-1] == 0:

                warnings.warn('Warning: step_size {} calculated is 0, setting up\
                              to 2, consider a reduction in the number of\
                              steps'.format(i))
                self.number_of_steps -= 1
                self.step_size[-1] =(int(2*(self.max[i] -\
                                         self.min[i])/(self.number_of_steps)))
                for j in range(i+1):
                    self.step_size[j] =(int(2*(self.max[j] -\
                                             self.min[j])/(self.number_of_steps))) 

                if self.number_of_steps == 2:
                    break

        for i in range(self.number_of_steps):
            frames = []
            for j in range(self.dim):
                frames.append( [self.min[j]+i*int(self.step_size[j]/2),
                    self.min[j] + i*int(self.step_size[j]/2) + self.step_size[j]] )
            self.slices.append(frames)

class EmbeddedSlicer(DataDomainSlicer):
    '''
      Slice domain according to a given Fater DomainSlice and a Mater
     DomainSlice. Fater and Mater slicers may be any DomainSlice object
     obeying the following conditions:
      - Mater slice will slice the given data domain as usual.
      - Fater slice will then slice each Mater slice as usual.
      - Optional:
        --recursive: if its true will further slice embedded with
        Father or Mother slicing scheme up to a deepness. Default to True.
        --recursive_generator: only if recursive is true, choose between Fater
        and Mater scheme to recursevely slice the embedded domains. Defaults to
        Fater.
            --recursive_depth: number of time slicer will try to generate
        sub-intervals from bigger intervals given from above, starting by the
        Fater Slice, the intervals will be generated acording to
        recursive_generator choosen. Defaults to 2.

    '''
    def __init__(
        self,
        data_domain,
        number_of_steps,
        frame_selection,
        frame_fraction,
        mater_slicer=LadderSlicer,
        fater_slicer=LadderSlicer,
        recursive=True,
        recursive_generator='Fater',
        recursive_depth=2
    ):
        super(EmbeddedSlicer, self).__init__(data_domain,
                                             number_of_steps,
                                             frame_selection=frame_selection,
                                             frame_fraction=frame_fraction)
        #check slicer objects
        if (fater_slicer.__name__ == 'EmbeddedSlicer'):
            raise WrongSlicerError(fater_slicer, ', use a non EmbeddedSlicer\
                                   as Fater Slicer')
        if (mater_slicer.__name__ == 'EmbeddedSlicer'):
            raise WrongSlicerError(mater_slicer, ', use a non EmbeddedSlicer\
                                   as Mater Slicer')

        if issubclass(fater_slicer,DataDomainSlicer) and\
            issubclass(mater_slicer,DataDomainSlicer):
            self.mater = mater_slicer(data_domain, number_of_steps,
                                      frame_selection, frame_fraction)
            self.fater = fater_slicer(data_domain, number_of_steps,
                                      frame_selection, frame_fraction)
            if recursive:
                self.generator = recursive_generator
                self.depth = recursive_depth
                self.recursive = recursive
        else:
            if not issubclass(fater_slicer,DataDomainSlicer):
                message = fater_slicer.__name__
            if not issubclass(mater_slicer,DataDomainSlicer):
                message = message+' and '+mater_slicer.__name__
            raise NotDomainSlicerError(message)

        self.configure(data_domain)

    def configure(self, data_domain):
        super( EmbeddedSlicer, self).configure(data_domain)

        if not self.mater._n == 0:
            self.mater.configure(data_domain)

        mater_slices = [i for i in self.mater]
        fater_slices = []
        if self.recursive:
            for deep in range(self.depth):
                for i in mater_slices:
                    self.slices.append(i)
                    self.fater.reset_domain(i)
                    self.fater.configure(i)
                    fater_slices.extend([j for j in self.fater])
                    self.slices.extend(fater_slices)
                mater_slices = fater_slices
                fater_slices = []
        else:

            for i in mater_slices:
                self.slices.append(i)
                self.fater.reset_domain(i)
                self.fater.configure(i)
                fater_slices.extend([j for j in self.fater])
                self.slices.extend(fater_slices)

        self.slices = np.unique(self.slices,axis=0)
        self.slices = self.slices.tolist()
        self.number_of_steps = len(self.slices)

class NoSliceSlicer(DataDomainSlicer):

    def __init__(self, data_domain,
                 number_of_steps=1,frame_selection='fraction',
                 frame_fraction=1.0):
        super(NoSliceSlicer, self).__init__(data_domain,
                                            number_of_steps,frame_selection,
                                            frame_fraction)
        self.configure(data_domain)

    def configure(self,data_domain):
        super(NoSliceSlicer, self).configure(data_domain)
        self.slices = [data_domain]

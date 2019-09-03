from template import NotDomainSliceError
import warnings
import numpy  as np

class DomainSlice(object):

    def __init__(self, setmax, setmin, number_of_steps):
        '''
         Abstract Class.
         Get domain bounds and set the domain slicing method.
        '''
        self.max=setmax
        self.min=setmin        
        self.number_of_steps = number_of_steps
        self.slices = []

    def configure(self):
        '''
         Configure the slicing based on the mode and the dimension supreme and
        infimum. In practice it returns a list of tuples, to attribute slices, each with the lower
        and the upper bound of the slice.

        abstract method to be implemented
        '''
        self.slices = [] 
        self._n = 0

    def set_max(self, setmax):
        if setmax > self.min:
            self.max = setmax

    def set_min(self, setmin):
        if setmin < self.max:
            self.min = setmin

    def reset_domain(self, setmin, setmax):
        self.max = 100000000000
        self.min = -1
        self.set_min(setmin)
        self.set_max(setmax)

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


class LadderSlice(DomainSlice):
    '''
      Slice the domain according to ladder-wise steps, that is, each step is
     half superpose with the previous step. Returns  steps according to the
     step_size and the given number of  steps
    '''

    def  __init__(self, setmax, setmin, number_of_steps):
        super(LadderSlice, self).__init__(setmax, setmin, number_of_steps)
        self.configure()

    def configure(self):
        '''
         Configure the slicing based on the mode and the dimension supreme and
        infimum. In practice it returns a list of tuples, to attribute slices, each with the lower
        and the upper bound of the slice.
        ''' 
        super( LadderSlice, self).configure()
        self.step_size = int(2*(self.max - self.min)/(self.number_of_steps+1))

        while self.step_size == 0:
            warnings.warn('Warning: step_size calculated is 0, will try to reduce the number of steps to get bigger domain size')
            self.number_of_steps -= 1
            self.step_size = int(2*(self.max - self.min)/self.number_of_steps)
            if self.number_of_steps <= 2:
                warnings.warn("Warning: cant find an adequate step size with \
                        number of steps bigger than 2, setting the both to 2,\
                        maybe is better to re-initialize with bigger domain \
                        size")
                self.step_size = 2
                break
        for i in range(self.number_of_steps):
            self.slices.append( ( self.min+i*int(self.step_size/2),
                self.min + i*int(self.step_size/2) + self.step_size) )

class EmbeddedSlice(DomainSlice):
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
    def __init__(self, setmax, setmin, number_of_steps, mater_slicer, fater_slicer,
            recursive=True, recursive_generator='Fater', recursive_depth=2):
        super(EmbeddedSlice, self).__init__(setmax, setmin, number_of_steps)
        #check slicer objects
        if isinstance(fater_slicer,DomainSlice) and isinstance(mater_slicer,DomainSlice): 
            self.mater = mater_slicer
            self.fater = fater_slicer            
            if recursive:
                self.generator = recursive_generator
                self.depth = recursive_depth
                self.recursive = recursive
        else:
            if not isinstance(fater_slicer,DomainSlice):
                message = fater_slicer.__class__
            if not isinstance(mater_slicer,DomainSlice):
                message = message+' and '+mater_slicer.__class__
            raise NotDomainSliceError(message)

        self.configure()

    def configure(self):
        super( EmbeddedSlice, self).configure()
        mater_slices = [i for i in self.mater]
        fater_slices = []
        if self.recursive:
            for deep in range(self.depth):
                for i in mater_slices:
                    self.slices.append(i)
                    self.fater.reset_domain(i[0],i[1])
                    self.fater.configure()
                    fater_slices.extend([j for j in self.fater])
                    self.slices.extend(fater_slices)
                mater_slices = fater_slices
                fater_slices = []
        else:

            for i in mater_slices:
                self.slices.append(i)
                self.fater.reset_domain(i[0],i[1])
                self.fater.configure()
                fater_slices.extend([j for j in self.fater])
                self.slices.extend(fater_slices)

        self.slices = np.unique(np.array(self.slices),axis=0)
        self.number_of_steps = len(self.slices)

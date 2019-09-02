
class DomainSlice(object):

    def __init__(self, setmax, setmin):
        '''
         Abstract Class.
         Get domain bounds and set the domain slicing method.
        '''
        self.max=setmax
        self.min=setmin
        self.slices = []

    def configure(self):
        '''
         Configure the slicing based on the mode and the dimension supreme and
        infimum. In practice it returns a list of tuples, to attribute slices, each with the lower
        and the upper bound of the slice.

        abstract method to be implemented
        '''
        pass

    def get_slice(self):
        '''
        Returns tuple with 2 elements corresponding to the lower and
        upper bound of the slice respectively. The number of bounds and the
        size  of the slice depends upon slice mode.

        abstract method to be implemented
        '''
        pass

    def __next__(self):
        return self.get_slice()

    def __iter__(self):
        return self


class LadderSlice(DomainSlice):
    '''
      Slice the domain according to ladder-wise steps, that is, each step is
     half superpose with the previous step. Returns  steps according to the
     step_size and the given number of  steps
    '''

    def  __init__(self, setmax, setmin, number_of_steps, step_size):
        super(LadderSlice, self).__init__(setmax, setmin)
        self.number_of_steps = number_of_steps
        self.step_size = step_size
        self.configure()

    def configure(self):
        '''
         Configure the slicing based on the mode and the dimension supreme and
        infimum. In practice it returns a list of tuples, to attribute slices, each with the lower
        and the upper bound of the slice.
        ''' 
        possible_steps_num = int((self.max - self.min)/ int(self.step_size/2)
                )-1

        if self.number_of_steps > possible_steps_num:
            self.number_of_steps = possible_steps_num

        for i in range(self.number_of_steps):
            self.slices.append( ( self.min+i*int(self.step_size/2),
                self.min + i*int(self.step_size/2) + self.step_size) )
        self._n = 0

    def get_slice(self):
        '''
         Returns tuple with 2 elements corresponding to the lower and
        upper bound of the slice respectively. The number of bounds and the
        size  of the slice depends upon slice mode.
        '''

        if self._n < self.number_of_steps:
            slic = self.slices[self._n]
            self._n += 1
            return slic
        else:
            raise StopIteration()

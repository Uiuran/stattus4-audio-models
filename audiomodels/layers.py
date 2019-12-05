from .builder import Builder
from .util import *
import warnings

class Signal(Builder):

    def __init__(
                 self,
                 dtype=None,
                 number_of_inputs=1,
                 channels=1,
                 name='signal_in'
    ):
        Builder.__init__(self, number_of_inputs=number_of_inputs,
                                      dtype=dtype , channels=channels,
                                      atype='placeholder')
        self.name=name
        self.namescopes.signal={}

    def __call__(self, input, tuning=None, **kwargs):
        Builder.__call__(self, input, **kwargs)

        if self.input is not 'dataset':
            raise ValueError('input for Signal layer must be a dataset.')
        if tuning is None:
            raise NoHyperparameterError('Signal Layer must hold the hyperparameter tuning of the network')

        self.data=input
        if self.number_of_inputs>tuning.slicer.number_of_steps:
            warnings.warn('The number of simultaneous Input Frames in the Signal Layer is greater than the number of Frames in the Hyperparameter Tuner.')
            self.number_of_inputs=tuning.slicer.number_of_steps

        while self.input_feeded<self.number_of_inputs:
            slice=tuning.slicer.get_slice()
            Builder.build_graph_module(self, 'placeholder', slice=slice, name=self.name)
            self.input_feeded += 1
        return self

    def feed_batch(self):
        '''
         Returns feed_dict of a batch of binary class data with one channel.

         TODO: implement for general labeled data
        '''
        data=self.data.training()
        self.feed_dict={}
        self.label=ExtendList([])

        for i in range(self.number_of_inputs):
            shape=ExtendList([2*len(data)])
            slice=self.namescopes.signal[i][0]
            for el in slice:
                shape.append(el[1]-el[0])
            shape.append(self.channels)
            shape=shape.to_tuple()

            for j in range(len(data)):
                d0=np.zeros(shape)
                d0[2*j,:,:,0]=data[j][0][2][slice[0][0]:slice[0][1],slice[1][0]:slice[1][1]]
                d0[2*j+1,:,:,0]=data[j][1][2][slice[0][0]:slice[0][1],slice[1][0]:slice[1][1]]
                self.label.append([0.0,1.0])
                self.label.append([1.0,0.0])
            self.feed_dict[self.namescopes.signal[i][1]]=d0

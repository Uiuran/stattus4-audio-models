# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python import util
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.client import timeline
from .util import *
from .template import BaseDataSampler
from .errors import UnknownArchBlockException, UnknownInputError

class Builder(object):

    def __init__(self, **kwargs):
        '''
         Template class to Build Architectural Blocks of Tensorflow computational graph.

        Requires signal input parameters such:
        - datasize: give a tuple with the signal input shape
        - channels: self explain.

        Note 0: You can always call Builder.get_directives(archblocktype) to know the inner params to the building block architecture.
        Note 1: The object guard reference for each op in a dictionary namescopes with the name of each block.
        '''

        self.namescopes = AttrDict()
        self._resolve_kwargs(**kwargs)
        self._deploy_architecture()

        self.graph = tf.get_default_graph()
        self.arch_blocks = {}

    def __call__(self, input, **kwargs):
        self.call(input, **kwargs)

    def call(self, input, **kwargs):
        for kw in kwargs:
            if kw == 'channels':
                self.channels=kwargs[kw]
        self._resolve_input(input)
        self.config_block_name(**kwargs)

    def config_block_name(self, **kwargs):
        '''
         Helper function to configure block architecture names.
         Inputs are deepness of the block, if there is any hidden layer, and the block number if there is more than one block with this architecture in the graph.
        '''
        allowed_kwargs={'numblock','deepness','pool'}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Key not allowed in this function', kwarg)
            else:
                if kwarg=='pool':
                    if kwargs[kwarg]<0:
                        self.pool = ['']
                        for i in range(kwargs[kwarg]):
                            self.pool.append('_p'+str(i))
                elif kwarg=='deepness':
                    if kwargs[kwarg]<0:
                        self.deepness = ['']
                    else:
                        self.deepness = ['']
                        for i in range(kwargs[kwarg]):
                            self.deepness.append('_d'+str(i))
                elif kwarg=='numblock':
                    if kwargs[kwarg]<=0:
                        if numblock <= 0:
                            self.numblock = ''
                        else:
                            self.numblock = '_'+str(numblock)
    def build_graph_module(self, arch, show_cgraph=False, verbose=False, **kwargs):
        '''
          Build a Tensorflow computational graph.building_function
          Currently supports 1-D gcnn or residual 1-D gcnn for unidimensional variable and 2D gcnn for bidimensional frame data.

          tensor_scope_name:  name scope of the architecture unit to build the next module

          kwargs:
              filter_size: conv filter size. Scalar number for time series, bidimensional tuple for image.
              channels_out: Number of channels for output convolution.
        '''
        self.architectures[arch](**kwargs)

        with self.graph.as_default():
            if show_cgraph:
                print_display_cgraph(self.graph, verbose=verbose)

    def assertive(self, archname):

        if archname == 'maxpooling2d':
            if hasattr(self, 'pool') != True or getattr(self, 'pool') == None:
                raise NameError('{} pools not setted'.format(archname))
            if hasattr(self, 'numblock') != True or getattr(self, 'numblock') == None:
                raise NameError('{} block number not setted'.format(archname))
        else:
            if hasattr(self, 'deepness') != True or getattr(self, 'deepness') == None:
                raise NameError('{} deepness not setted'.format(archname))
            if hasattr(self, 'numblock') != True or getattr(self, 'numblock') == None:
                raise NameError('{} block number not setted'.format(archname))

    def adaptative_learning_rate(self,step,K):

        _ff = lambda ord,loss : 10.0*ord if loss//ord < 1.0/ord else ord
        _f = lambda ord,loss : ord/10.0 if loss//ord > 1.0/(ord/10.0) else ord
        f = lambda ord,loss: _f(ord,loss) if _ff(ord,loss) == ord else _ff(ord,loss)
        ord = np.mean( np.abs( np.diff( self.lossval[step-K:step] ) ) )
        ord = f(ord, self.lossval[step])
        lr = ( ( ( np.log2(step*0.01+2.0)*self.lr)**(1./(np.log2(step*0.1+1.0)+1.0)))/( ( self.lossval[step]//(self.lossval[step]*0.05) )*(1.0 + step*(ord//(0.002) ) ) ) )%0.001
        return (lr + (lr // 0.0000005) )% 0.0003 + 0.0000005

    def get_directives(self,arch):
        return self.directives[arch]

    def _deploy_architecture(self):

        self.architectures = AttrDict()
        if self.block=='gcnn2d':
            self.architectures.gcnn2d=self._gcnn2d
        elif self.block=='reducemean':
            self.architectures.reducemean=self._reducemean
        elif self.block=='losscrossentropy':
            self.architectures.losscrossentropy=self._losscrossentropy
        elif self.block=='softmax':
            self.architectures.softmax=self._softmax
        elif self.block=='maxpooling2d':
            self.architectures.maxpooling2d=self._maxpooling2d
        elif self.block=='cnn2d':
            self.architectures.cnn2d=self._cnn2d
        elif self.block=='placeholder':
            self.architectures.placeholder=self._placeholder
        self.directives = {
            'gcnn2d': {'channels_out':1,
                       'filter_size':2},
            'reducemean':{'None':None},
            'losscrossentropy':{'num_labels':1,
                                'batch':1,
                                'learningrate':1},
            'softmax':{'num_labels':1},
            'residualgcnn1d':{'channels_out':1,
                              'filter_size':1},
            'gcnn1d':{'channels_out':1,
                      'filter_size':2
                     },
            'cnn2d':{'channels_out':1,
                     'filter_size':2
                    },
            'maxpooling2d':{'poolsize':2
            },
            'signalimg':{'batch':1,
                         'datasize':2,
                         'channels':1,
                         'dtype':1
                        }
        }

    def _placeholder(self, **kwargs):

        if "name" in kwargs:
            name=kwargs['name']
        else:
            name='signal'
        if "slice" in kwargs:
            slice=kwargs['slice']
        else:
            raise ValueError('input slice not found.')

        shape=ExtendList([None])
        for el in slice:
            shape.append(el[1]-el[0])
        shape.append(self.channels)
        shape=shape.to_tuple()

        with self.graph.as_default():
            with tf.variable_scope(name+'/'):
                self.namescopes.signal[self.input_feeded]=[slice,tf.placeholder(
                        self.dtype,
                        shape=shape,
                        name='signal_in_'+str(self.input_feeded)
                    )]

    def _gcnn2d(self, **kwargs):
          try:
              channels_out = kwargs['channels_out']
              filter_size = kwargs['filter_size']
          except KeyError:
              sys.exit('Parameters Not Defined Error')

          signal_in = self(**kwargs)
          self.assertive('gcnn2d')

          with self.graph.as_default():

              with tf.variable_scope('gcnn2d'+self.numblock+self.deepness[0]):

                  self.namescopes['gcnn2d'+self.numblock+self.deepness[0]] = []
                  with self.graph.device(dev_selector(arg1='foo')('gcnn2d')):
                      conv_linear = tf.keras.layers.Conv2D( channels_out, filter_size, padding='valid', name='conv_linear', use_bias=True, kernel_initializer=tf.initializers.lecun_normal(seed=137), bias_initializer=tf.initializers.lecun_normal(seed=137) )(signal_in)
                      self.namescopes['gcnn2d'+self.numblock+self.deepness[0]].append(conv_linear)

                  with self.graph.device(dev_selector(arg1='foo')('gcnn2d')):
                      conv_gate = tf.sigmoid(tf.keras.layers.Conv2D(
                          channels_out, filter_size, padding='valid',
                          name='conv', use_bias=True,
                          kernel_initializer=tf.initializers.lecun_normal(seed=137),
                          bias_initializer=tf.initializers.lecun_normal(seed=137)
                          )(signal_in),name='conv_sigmoid')
                      self.namescopes['gcnn2d'+self.numblock+self.deepness[0]].append(conv_gate)

                  with self.graph.device(dev_selector(arg1='foo')('gcnn2d')):
                          gated_convolutions = tf.multiply(conv_linear,conv_gate,name='gated_convolutions')
                          self.namescopes['gcnn2d'+self.numblock+self.deepness[0]].append(gated_convolutions)

                  self.deepness.pop(0)
                  if len(self.deepness) == 0:
                      self.deepness = None

    def _cnn2d(self, **kwargs):
        try:
            channels_out = kwargs['channels_out']
            filter_size = kwargs['filter_size']

        except KeyError:
            sys.exit('Parameters Not Defined Error')

        signal_in = self(**kwargs)
        self.assertive('cnn2d')

        with self.graph.as_default():
            with tf.variable_scope('cnn2d'+self.numblock+self.deepness[0]):

                self.namescopes['cnn2d'+self.numblock+self.deepness[0]] = []
                with self.graph.device(dev_selector(arg1='foo')('cnn2d')):
                    conv_linear = tf.keras.layers.Conv2D( channels_out, filter_size, padding='valid', name='conv_linear', use_bias=True, kernel_initializer=tf.initializers.lecun_normal(seed=137), bias_initializer=tf.initializers.lecun_normal(seed=137), activation=None )(signal_in)
                    lrelu = tf.keras.layers.LeakyReLU(alpha=0.3)(conv_linear)
                    self.namescopes['cnn2d'+self.numblock+self.deepness[0]].append(lrelu)

                self.deepness.pop(0)
                if len(self.deepness) == 0:
                    self.deepness = None

    def _maxpooling2d(self, **kwargs):

        try:
            poolsize = kwargs['poolsize']
        except:
            poolsize = (2,2)

        signal_in = self(**kwargs)
        self.assertive('maxpooling2d')
        with self.graph.as_default():

            with tf.variable_scope('maxpooling2d'+self.numblock+self.pool[0]): 
                self.namescopes['maxpooling2d'+self.numblock+self.pool[0]] = []
                maxpool = tf.keras.layers.MaxPool2D( pool_size=poolsize, strides=None, padding='valid', data_format=None)(signal_in)
                self.namescopes['maxpooling2d'+self.numblock+self.pool[0]].append(maxpool)

            self.pool.pop(0)
            if len(self.pool) == 0:
              self.pool = None


    def _reducemean(self, **kwargs):

#        scope_tensor_name = find_softmax(self.graph)
#        inputs = get_tensor_list_from_scopes(self.graph, scope_tensor_name)
        self.assertive('reducemean')
        with self.graph.as_default():
            with tf.variable_scope('reducemean'+self.numblock+self.deepness[0]):

                scope_tensor_name = find_softmax(self.graph)
                inputs = get_tensor_list_from_scopes(self.graph, scope_tensor_name)
                self.namescopes['reducemean'+self.numblock+self.deepness[0]] = []
                with self.graph.device(dev_selector(arg1='foo')('reducemean')):
                    rm = tf.math.reduce_mean( inputs, axis=1 )
                    self.namescopes['reducemean'+self.numblock+self.deepness[0]].append(rm)
                    rmshape = tf.shape(rm)
                    rmean = tf.reshape(rm, shape=(rmshape[0], rmshape[2]))
                    self.namescopes['reducemean'+self.numblock+self.deepness[0]].append(rmean)

                self.deepness.pop(0)
                if len(self.deepness) == 0:
                    self.deepness = None

    def _softmax(self, **kwargs):

        try:
            num_labels = kwargs['num_labels']
        except Exception:
            sys.exit('Parameters Not Defined Error')

        signal_in = self(**kwargs)
        self.assertive('softmax')

        with self.graph.as_default():

            with tf.variable_scope('softmax'+self.numblock+self.deepness[0]):
                self.namescopes['softmax'+self.numblock+self.deepness[0]] = []

                with self.graph.device(dev_selector(arg1='foo')('softmax')):

                    logits = tf.contrib.layers.fully_connected(signal_in, num_labels, activation_fn=None, normalizer_fn=None, normalizer_params=None, weights_initializer=tf.initializers.lecun_normal(seed=731), weights_regularizer=None, biases_initializer=tf.initializers.lecun_normal(seed=777), biases_regularizer=None, reuse=None, variables_collections=None, outputs_collections=None, trainable=True, scope='logit')

                    logitrank = tf.rank(logits,name='logit'+self.numblock+self.deepness[0])
                    self.namescopes['logit'+self.numblock+self.deepness[0]] = logitrank
                    self.namescopes['softmax'+self.numblock+self.deepness[0]].append(logits)

                with self.graph.device(dev_selector(arg1='foo')('softmax')):
                    with tf.control_dependencies([
                        tf.Assert(
                            tf.equal(logitrank,4),
                            [logitrank,logits])
                        ]):

                        shape = np.shape(logits)
                        convolved_logit = tf.cond( tf.logical_or( tf.not_equal(shape[1], 1), tf.not_equal(shape[2], 1) ) , lambda:
                                     self._matchconv(logits),
                                     lambda: logits)

                    softmax = tf.nn.softmax(convolved_logit,axis=0)
                    self.namescopes['softmax'+self.numblock+self.deepness[0]].append(softmax)

                self.deepness.pop(0)
                if len(self.deepness) == 0:
                    self.deepness = None

    def _matchconv(self,logit):

        k1= logit.get_shape()[1].value
        k2= logit.get_shape()[2].value
        return tf.keras.layers.Conv2D( 2, (k1, k2), padding='valid',
            name='matchconv', use_bias=True,
            kernel_initializer=tf.initializers.lecun_normal(seed=137),
            bias_initializer=tf.initializers.lecun_normal(seed=137),
            activation=None )(logit)

    def _losscrossentropy(self, **kwargs):

        try:
            num_labels = kwargs['num_labels']
            batch = 10 #kwargs['batch']
        except Exception:
            sys.exit('Parameters Not Defined Error')

        signal_in = self(**kwargs)
        self.assertive('losscrossentropy')

        with self.graph.as_default():

            self.learning_rate = tf.placeholder(tf.float32,shape=(),name='learning_rate')
            with tf.variable_scope('losscrossentropy'+self.numblock+self.deepness[0]):

                self.namescopes['losscrossentropy'+self.numblock+self.deepness[0]] = []
                self.label_tensor = tf.placeholder(tf.float32,(None,num_labels), name='labels')

                with self.graph.device(dev_selector(arg1='foo')('losscrossentropy')):
                    self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0)(self.label_tensor, signal_in)
                    self.namescopes['losscrossentropy'+self.numblock+self.deepness[0]].append(self.loss)


                with self.graph.device(dev_selector(arg1='foo')('losscrossentropy')):
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    self.minimize_op = self.optimizer.minimize(self.loss)
                self.namescopes['losscrossentropy'+self.numblock+self.deepness[0]].append(self.optimizer)
                self.namescopes['losscrossentropy'+self.numblock+self.deepness[0]].append(self.minimize_op)

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            self.deepness.pop(0)
            if len(self.deepness) == 0:
                self.deepness = None

    def _resolve_kwargs(self, **kwargs):

        allowed_kwargs={'channels','dtype','number_of_inputs','name','atype'}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Key not allowed in this function', kwarg)
            else:
                if kwarg=='channels':
                    self=Util.set_with_assert(self, 'channels', kwargs[kwarg], int)
                elif kwarg=='dtype':
                    self=Util.set_with_assert(self, 'dtype', kwargs[kwarg], type)
                elif kwarg=='number_of_inputs':
                    self=Util.set_with_assert(self, 'number_of_inputs', kwargs[kwarg], int)
                    self.input_feeded=0
                elif kwarg=='atype':
                    self=Util.set_with_assert(self, 'block', kwargs[kwarg], str)
                    if Util.all(self.block !=
                                ExtendList(
                                    ['gcnn2d',
                                     'cnn2d',
                                     'maxpooling2d',
                                     'softmax',
                                     'reducemean',
                                     'losscrossentropy',
                                     'matchconv',
                                     'flatten',
                                     'placeholder']
                                )
                               ):
                        raise UnknownArchBlockException(self.block)

    def _resolve_input(self, input):

        if Util.assert_class(input.__class__, Builder):
            if input.__name__=='Signal':
                self.input='signaling'
            if input.__name__=='DeepBlock':
                #TODO
                pass
        elif Util.assert_class(input.__class__, str):
            self.input='namescope'
        elif Util.assert_class(input.__class__, BaseDataSampler):
            self.input='dataset'
        else:
            raise UnknownInputError(input)

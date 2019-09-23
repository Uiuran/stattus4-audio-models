# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python import util
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.client import timeline

from spect import *
from cgraphviz import *
from tfhelper import *
from dataloader import *
from ckpt_hookers import *
from domain import *
from hyperparamtuning import  *
from util import *

# SHORT NAMES
tf.summary.initialize = tf.contrib.summary.initialize
tf.variable_scope = tf.compat.v1.variable_scope

# TODO- json file is sourcing the argments to the objects,
# in the case of given config file. Possibly do the option to build kwargs
# with objects of data/framing and arch_search from inside the model, without
# .json file.
class Model:
    '''
      Model Class uses Builder class to build each architecture block of the
     network and build the final model that will be trained.

     Argments

        -data: BaseDataSampler instance with the data_path and other argments
        relevant to data reading, see BaseDataSampler and its heritance tree
        doc-string.

        -arch_search: Hyperparameter instance, holds the slicer object and the
        tuning algorithm, see the docstring for its argments.

        -config: string with path to .json file that configures all relevant
        argments. If it is given, it will have priority to build the model over
        the given data and arch_search.

    Kwargs:

        Optionally configure objects data and arch_search given not as a
        instance but as string selector.

        **Not implemented, design decision for the future**

    '''
    def __init__(
        self,
        data=None,
        arch_search=None,
        config=None,
        verbose=None,
        **kwargs
    ):

        # Give preference to config kwarg that gives everything in a json file
        if config is not None:
            self.config=None
            self._set_with_assert('config', config, str)
            # assert it is json config file
            if self.config.find('.json') is not -1:
                self.read_from_json()
                self.configure()
            else:
                raise('config keyword arg must be a string with full path to a valid .json file')
        else:
            if self._assert_instance(data, BaseDataSampler):
                self.data=data
            else:
                raise ValueError("data must be an instance of BaseDataSampler.")
            if self._assert_instance( arch_search, Hyperparameter):
                self.arch_search=arch_search
            else:
                raise ValueError("arch_search must be a instance of Hyperparameter")

        if verbose:
            self.verbose=True

    def _set_with_assert(self,holder, argment, classtype):
        '''
         Helper function to assert argments
        '''
        if issubclass(argment.__class__, classtype):
            # instance assertion
            if hasattr(self, holder):
                self.__dict__[holder]=argment
        elif issubclass(argment.__class__,type):
                    # class assertion
            if issubclass(argment, classtype):
                if hasattr(self, holder):
                    self.__dict__[holder]=argment
                else:
                    raise ValueError(str(argment.__class__.__name__)+" must be a "+classtype.__name__+" class to be configured")
        else:
            raise ValueError(str(argment.__class__.__name__)+" must be a "+classtype.__name__+" instance to be configured")

    @staticmethod
    def _assert_instance(argment, classtype):
        if issubclass(argment.__class__, classtype):
            # instance assertion
            return True
        else:
            return False

    @staticmethod
    def _assert_class(argment, classtype):

        if issubclass(argment.__class__,type):
                    # class assertion
            if issubclass(argment, classtype):
                return True
            else:
                return False
        else:
            return False

    def print_namescopes(self, mode='std_out'):
        '''
        printar buider.namescopes
        '''
        if mode == 'std_out':
            print(self.builder.namescopes.items())

    def configure(self,from_json=False):

        if self.data_loader:
            # Data Loader initializators
            if issubclass(self.data_loader, Sass):
                self.data_loader=self.data_loader(
                    self.data_path,
                    **self.data
                )
        if self.arch_search:
            if issubclass(self.slicer,EmbeddedSlicer):
                self.arch_search=self.arch_search(
                    self.data_domain,
                    **self.framing
                )
    def read_from_json(self):
        '''
         Configure model argments from .json file.
         Json attributes are based in the three basic objects that builds the
         model:

             - data: contains the data loader.
             - framing: contains the data domain structure slicer (frame
             generator).
             - arch_search: the hyperparameter searching algorithm, receive
             argments from framing to build.

        Each field returns a dictionary with its name to serve as argment in
        instance building. For non keyword or default valued argment the
        function returns an attribute with the name of the argment.

        Fields Descriptions:

        **data**

            Returned as Attribute of the Model instance.

            -data_path: string with full path to data repository.
            -name: Name of the data loader.  Returned as data_loader to be instantiated.

            Returned as keys of data dictionary of the Model instance

            -num_samples: number of samples per training and testing batching.
            -number_of_batches: number of batches per training and testing.
            -split_tax: how many data to split for training and testing.

        **framing**

            Returned as Attribute of the Model instance. These attributes are
            used as argment and parameters of Hyperparameter object.

            -data_domain: list of lists containing the bounds of the
            dimensions, transformed to tuple on purpose of hashing.
            -number_of_steps: number of frames to be generated by the slicer.
            -frame_selection: mode of selection of the generated frames,
            default is 'fraction', optionally 'all'.
            -frame_fraction: number between 0.0 and 1.0, the fraction of frames
            to be selected in the case of the frame_selection be 'fraction'.
            -name: name of the class to be instantied. Returned as slicer
            attribute.

            Returned as keys of framing dictionary of the Model Instance.

            -fater_slicer: Slicer to be used by EmbeddedSlicer.
            -mater_slicer: Slicer to be used by EmbeddedSlicer
            -recursive: boolean, if true try to slices the frames to generate
            new frames.
            -recursive_depth: int, number of times that new frames are
            generated according to the recursive_generator Slicer. Defaults to
            2.
            -recursive_generator: basis Slicer to recursively slice frames and
            generate new frames. Defaults to 'Fater' to use fater_slicer.

        **arch_search**

            Returns arch_search as attribute to instantiate Hyperparameter
            object from framing field.

        '''

        try:
            with open(self.config,'r') as fp:
                config = json.load(fp)
        except:
            raise ValueError('invalid pathname to json file')

        #TODO - do assertions for data in the place of pass.
        if config.has_key('data'):
            self.data=config['data']
            if self.data.has_key('data_path'):
                self.data_path=self.data.pop('data_path')
            else:
                raise KeyError('data_path key not found.')
            if self.data.has_key('num_sampĺes'):
                pass
            else:
                self.data['num_samples']=10
            if self.data.has_key('number_of_batches'):
                pass
            else:
                self.data['number_of_batches']=4
            if self.data.has_key('split_tax'):
                pass
            else:
                self.data['split_tax']=0.2
            if self.data.has_key('name'):
                #Data-loader types reader
                name=self.data.pop('name')
                if name=='Sass' or\
                    name=='Stattus4AudioSpectrumSampler':
                    self.data_loader = Sass
                    if self.data.has_key('freq_size'):
                        pass
                    else:
                        self.data['freq_size']=200
                    if self.data.has_key('time_size'):
                        pass
                    else:
                        self.data['time_size']=200
                    if self.data.has_key('same_data_validation'):
                        pass
                    else:
                        self.data['same_data_validation']=True
        else:
            raise KeyError('no data field, in json configuration file')
        if config.has_key('framing'):
            self.framing=config['framing']
            if self.framing.has_key('data_domain'):
                if issubclass(self.framing['data_domain'].__class__,list):
                    if issubclass(self.framing['data_domain'][0].__class__,list):
                        self.data_domain=tuple([tuple(a) for a in self.framing.pop('data_domain')])
                    else:
                        raise Exception('dimension bounds are not properly setted')
                else:
                    raise Exception('domain must be a list')
            else:
                raise KeyError('.json does not have data_domain list object')

            if self.framing.has_key('number_of_steps'):
                self.number_of_steps=self.framing.pop('number_of_steps')
            else:
                self.number_of_steps=10
            if self.framing.has_key('frame_selection'):
                self.frame_selection=self.framing.pop('frame_selection')
            else:
                self.frame_selection='fraction'
            if self.framing.has_key('frame_fraction'):
                self.frame_fraction=self.framing.pop('frame_fraction')
            else:
                self.frame_fraction=0.15
            if self.framing.has_key('name'):
                name=self.framing.pop('name')
                if name=='EmbeddedSlicer':
                    self.slicer=EmbeddedSlicer
                    if self.framing.has_key('fater_slicer'):
                        pass
                    else:
                        self.framing['fater_slicer']=LadderSlicer
                    if self.framing.has_key('mater_slicer'):
                        pass
                    else:
                        self.framing['mater_slicer']=LadderSlicer
                    if self.framing.has_key('recursive'):
                        pass
                    else:
                        self.framing['recursive']=True
                    if self.framing.has_key('recursive_depth'):
                        pass
                    else:
                        self.framing['recursive_depth']=2
                    if self.framing.has_key('recursive_generator'):
                        pass
                    else:
                        self.framing['recursive_generator']="Fater"
                if name == 'LadderSlicer':
                    self.slicer=LadderSlicer
                if name == 'NoSliceSlicer':
                    self.slicer=NoSliceSlicer
        if config.has_key('arch_search'):
            self.arch_search=config['arch_search']
            if self.arch_search=='GCNNMaxPooling':
                self.arch_search=GCNNMaxPooling
        else:
            self.arch_search=GCNNMaxPooling

    def info(self):
        print(self.__dict__)

    def feed_batch(self):
        '''
         Feed a batch to the model according to the configured data-loader.
        '''
        pass

    def compile(self):
        '''
         Join the pieces of the Model together:

             1- Signal Input as it multiple Frame Manifolds.
                -- Slicer Module and Builder with Signal Input Module.

             2- Deep Architecture Pieces for each Manifold and each Frame.
                -- Hyperparameter Module implementing the NAS Algorithms (Neural Architecture Search).

             3- Dataloader and Checkpointing.
                -- Checkpointing with Training Evaluation Module.
        '''


class Builder:

    def __init__(self, **kwargs):
        '''
         Build a Tensorflow computational graph, from scratch.

        Requires signal input parameters such:
        - datasize: give a tuple with the signal input shape
        - channels: self explain.

        Note 0: You can always call Builder.get_directives(archblocktype) to know the inner params to the building block architecture.
        Note 1: The object guard reference for each op in a dictionary namescopes with the name of each block.
        '''

        self.namescopes = AttrDict()
        self.architectures = AttrDict()
        self.architectures.gcnn2d = self._gcnn2d
        self.architectures.reducemean = self._reducemean
        self.architectures.losscrossentropy = self._losscrossentropy
        self.architectures.softmax = self._softmax
        self.architectures.maxpooling2d = self._maxpooling2d
        self.architectures.cnn2d = self._cnn2d

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

        tf.reset_default_graph()

        try:

          self.datasize = kwargs['datasize']
          self.channels = '' or kwargs['channels']
          self.dtype = kwargs['dtype']

        except KeyError:

            sys.exit('Signal Input Not Defined Error')

        try:

            self.num_input = kwargs['num_input']

        except:

            self.num_input = 1

        self.graph = tf.Graph()
        self.signal_in = ExtendList([])
        self.arch_blocks = {}

    def __call__(self, **kwargs):

        try:
            isinput = kwargs['isinput']
            if type(isinput) != type(bool()):
                isinput = True
        except:
            isinput = False

        try:
            lastnamescope= kwargs['lastnamescope']
            if type(lastnamescope) != type(bool()):
                lastnamescope= False
        except:
            lastnamescope= True


        if isinput:
            with self.graph.as_default():
              # TODO - Different input shapes and data types (char,string,hexa,float,int...)
                if self.num_input >= 0:
                    self.signal_in.append(tf.placeholder(self.dtype,shape=(None,
                        self.datasize[0][self.num_input-1],
                        self.datasize[1][self.num_input-1], self.channels), name='signal_in'))
                    self.namescopes['signal_in'] = self.signal_in
                    self.num_input -= 1
                    return self.signal_in[-1]
        elif lastnamescope:
            return self.graph.get_tensor_by_name(self.graph.get_operations()[-1].name+':0')

        else:
            try:
                namescope = kwargs['namescope']

                with self.graph.as_default():
                    print('ok')
                    with tf.name_scope(namescope+'/'):
                        opa = 0
                        for op in self.graph.get_operations():
                            if op.name.split('/')[0] == namescope:
                                opa = op
                        return self.graph.get_tensor_by_name(opa.name+':0')
            except:
                sys.exit("Namescope not given to the input of architectures")

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

    def adaptative_learning_rate(self,step,K):

        _ff = lambda ord,loss : 10.0*ord if loss//ord < 1.0/ord else ord
        _f = lambda ord,loss : ord/10.0 if loss//ord > 1.0/(ord/10.0) else ord
        f = lambda ord,loss: _f(ord,loss) if _ff(ord,loss) == ord else _ff(ord,loss)
        ord = np.mean( np.abs( np.diff( self.lossval[step-K:step] ) ) )
        ord = f(ord, self.lossval[step])

        lr = ( ( ( np.log2(step*0.01+2.0)*self.lr)**(1./(np.log2(step*0.1+1.0)+1.0)))/( ( self.lossval[step]//(self.lossval[step]*0.05) )*(1.0 + step*(ord//(0.002) ) ) ) )%0.001
        return (lr + (lr // 0.0000005) )% 0.0003 + 0.0000005

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

    def get_directives(self,arch):

    # TODO: Use as automatic requirement builder from file, maybe json.
    # Provisory setup for namespace tags
    #self._nametags = {arch:{}}
        return self.directives[arch]

  # provisory configuration for next sequence of build module names, FUTURE: Hyper-Param-Protocol will learn
  # the names automatically from given configuration or from the graph architecture updating.
    def config_block_name(self, deepness=0, numblock=0, pool = -1):
        '''
         Helper function to configure block architecture names.
         Inputs are deepness of the block, if there is any hidden layer, and the block number if there is more than one block with this architecture in the graph.
        '''
        if pool < 0:
            self.pool = ['']
        else:
            self.pool = ['']
            for i in range(pool):
                self.pool.append('_p'+str(i))

        if deepness < 0:
            self.deepness = ['']
        else:
            self.deepness = ['']
            for i in range(deepness):
                self.deepness.append('_d'+str(i))
        if numblock <= 0:
            self.numblock = ''
        else:
            self.numblock = '_'+str(numblock)
  # Definitions: Build computation graph and run session
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

    def run_cgraph(self, feed_dict, op_to_run = '', number_of_runs = 10, mode =
          'minimize', new_session=True, ckpt_dir_name='./ckpt/model',
          output_log = True, adaptative_lr=True, k=3, stop=0.0001,
          verbose=False):
        '''
         Run Session. 
         Returns a list of tuples with values of output tensors passed to op_to_run. If op_to_run = '' (standard) does not evaluate any operation, only do the optimization.
         If mode = 'minimize' (standard) the minimizer will backprop the gradients calculated with the feedeed batch of data. It will save the model automatically if the loss function is lower than stop argment.
        '''

        if new_session:
            session = tf.Session(graph=self.graph)
          #session = tf_debug.TensorBoardDebugWrapperSession(session,"grpc://localhost:6064")
        else:
            session = tf.get_default_session()

        with session.as_default():

            feed_dict = feed_dict
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            opval = []
            self.lossval = np.zeros(number_of_runs)

        session.run(self.init_op)
        print('Global Variables Initialized')
        self.op_to_run = op_to_run
      #with tf.summary.FileWriter('/home/penalvad/stattus4/stattus4-audio-models/notebooks/',graph=self.graph,session=session) as writer:
        #tf.summary.initialize(graph=self.graph,session=session)

        for step in range(number_of_runs):
            if mode == 'minimize':

                loss_value, learning_rate,_ = session.run([self.loss, self.learning_rate, self.minimize_op], feed_dict=feed_dict)
                self.lossval[step] = loss_value

                if step == 0:
                    self.lr = learning_rate

                if step % 1 == 0:
                    print('\n')
                    print("Step:", step, " Loss:", self.lossval[step])
                    print('\n')

                if step >= k and adaptative_lr:

                    if self.lossval[step] < self.lossval[step-1] and self.lossval[step] < stop:
                        print('salvando modelo...')
                        print('\n')
                        self.saver.save(session, ckpt_dir_name)

                    feed_dict[self.learning_rate]= self.adaptative_learning_rate(step,k)

                    if verbose:
                        print('\n')
                        print('learning rate ', self.adaptative_learning_rate(step,k) )
                        print('\n')

                elif step >= k:

                    if self.lossval[step] < self.lossval[step-1] and self.lossval[step] < stop:
                        print('salvando modelo...')
                        print('\n')
                        self.saver.save(session, ckpt_dir_name)

                    feed_dict[self.learning_rate]= self.lr

                if self.op_to_run != '':
                    op_values = session.run(self.op_to_run, feed_dict=feed_dict)
                    opval.append( op_values )

                    if verbose:
                        print('\n')
                        print(op_values)
                        print('\n')

        return opval

# Restoring Checkpointed Model by loading model meta data (model.meta), and rebuilding it from lastest Checkpoint (which is referenced in checkpoint file that is together with map file model.index and variables data file model.data-....)

def re_feed(sess, data_train, ltrain):

    feed_dict = {}
    feed_dict[sess.graph.get_tensor_by_name('signal_in:0')] = data_train[:4,:,:8,:]

    for i in np.arange(1,10):
        feed_dict[sess.graph.get_tensor_by_name('signal_in_'+str(i)+':0')] = data_train[:4,:,8*i:8*(i+1),:]
    feed_dict[sess.graph.get_tensor_by_name('losscrossentropy/labels:0')] = ltrain[:4]
    feed_dict[sess.graph.get_tensor_by_name('learning_rate:0')] = 0.0001

    return feed_dict

def restore_model(data_train, ltrain, model_dir_name = './ckpt/model'):
    '''
     Function returns reloaded computational graph
    '''
    graph = tf.Graph()
    with graph.as_default():

        saver = tf.train.import_meta_graph(model_dir_name+".meta", import_scope='')

        with tf.Session(graph=graph) as sess:

    # Restore variables from disk.

            saver.restore(sess, tf.train.latest_checkpoint('./ckpt/') )
            feed_dict = re_feed(sess, data_train, ltrain)

      #for op in graph.get_operations():
      #print(2*'\n')
      #print(op.name)
            minimize_op = sess.graph.get_operation_by_name('losscrossentropy/Adam')
            loss = sess.graph.get_tensor_by_name('losscrossentropy/categorical_crossentropy/weighted_loss/value:0')
            acc = sess.graph.get_tensor_by_name('reducemean/Mean:0')

    return graph, feed_dict, minimize_op, loss, acc

# Specific Exception Handler for the Input Datasets, not to be used yet

class Signal:

    def __init__(self, graph, data_shape, signaling_shape, dtype):
        self.signalin = []
        self.dshape = data_shape
        self.sshape = signaling_shape
        self.dtype = dtype

        with graph.as_default():
            with tf.name_scope('signal'):
                self.data_feed = tf.placeholder(dtype, shape=(data_shape[0], data_shape[1], data_shape[2], data_shape[3]), name='data_feed')

    def feed_signal_in():
        pass

    def signal_in(self, graph):

        with graph.as_default():
            with tf.name_scope('signal/'):
      # TODO - Different input shapes and data types (char,string,hexa,float,int...)                   
                self.signalin.append( tf.placeholder(self.dtype,shape=(self.sshape[0], self.sshape[1], self.sshape[2], self.sshape[3]), name='signal_in') )       

    def __call__(self, index = -1):
        return self.signal_in[index]

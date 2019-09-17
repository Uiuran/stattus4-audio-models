# encoding : utf-8

# IMPORTS
from IPython.display import Math
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

import numpy as np
import matplotlib.pyplot as plt
from six import string_types
import argparse
import functools
from numpy import asarray, array, ravel, repeat, prod, mean, where, ones
import sys
import copy
import re
import os

# SHORT NAMES
tf.summary.initialize = tf.contrib.summary.initialize
tf.variable_scope = tf.compat.v1.variable_scope

# CONVENIENCE DATA STRUCTS
class AttrDict(dict):

  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

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
        self.signal_in = []
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

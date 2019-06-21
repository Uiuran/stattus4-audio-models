# -*- coding: utf-8 -*-
"""Audio_GatedCNNArch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_RlpG-FlBbsc2DT9Wm3Cc3-ALcPHhx0_
"""

from IPython.display import Math
!pip install librosa
import tensorflow as tf
import numpy as np
import scipy as scp
from scipy import signal, linalg
from numpy import asarray, array, ravel, repeat, prod, mean, where, ones
!pip install scikits.audiolab
!pip install --upgrade-strategy=only-if-needed git+https://github.com/Uiuran/BregmanToolkit
!pip install scikit-image
import scikits.audiolab as audio
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
correlate = signal.correlate
import os
# Tensorboard display
from IPython.display import clear_output, Image, display, HTML
'''
GATED
CONVOLUTIONAL NEURAL NETWORK Architecture P.O.C (proof of concept)
'''

"""<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><li><span><a href="#funcdef" data-toc-modified-id="funcdef-0"><span class="toc-item-num">0&nbsp;&nbsp;</span>Function Definitions</a></span></li><a href="#archbasic" data-toc-modified-id="archbasic-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Architecture Basics</a></span></li><li><span><a href="#Data-Pre-processing-and-Statistical-Analysis" data-toc-modified-id="Data-Pre-processing-and-Statistical-Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Pre-processing and Statistical Analysis</a></span><ul class="toc-item"><li><span><a href="#1.-Load-Data-,-drop/input-value-in-NaN." data-toc-modified-id="1.-Load-Data-,-drop/input-value-in-NaN.-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>1. Load Data , drop/input value in NaN.</a></span></li><li><span><a href="#1.1-Data-Mocking:-Langevin-Integration" data-toc-modified-id="1.1-Data-Mocking:-Langevin-Integration-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>1.1 Data Mocking: Langevin Integration</a></span></li><li><span><a href="#2.-General-Cleaning:-Log-diff-(for-non-stationary-time-series)-,-0.0-values-interpolation,-Standardization." data-toc-modified-id="2.-General-Cleaning:-Log-diff-(for-non-stationary-time-series)-,-0.0-values-interpolation,-Standardization.-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>2. General Cleaning: Log-diff (for non-stationary time-series) , 0.0 values interpolation, Standardization.</a></span><ul class="toc-item"><li><span><a href="#2.1-Data-Serialization-(json)" data-toc-modified-id="2.1-Data-Serialization-(json)-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>2.1 Data Serialization (json)</a></span></li></ul></li></div>
"""

# funções auxiliares
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
# Função que usa HTML e javascript para exibir tensorboar no notebook e web
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
    
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.    
    
    Parameters:
    
    - session: The TensorFlow session to be frozen.
    
    Default Keyword Parameters
    
    - keep_var_names: A list of variable names that should not be frozen. Defaults None.     
    - output_names: Names of the relevant graph outputs/operation/tensor to be written. Defaults None.
    - clear_devices: Remove the device directives from the graph for better portability. Defaults True.
    
    return The frozen graph definition.
    """    
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def sample_gaussmix(data=None,shape=(500,), mean_base = 0.0,var_base = 0.15, num_mods = 30, random_mods=False):
  '''
    Generate sample noise from num_mods different gaussian with data or only noise.
    
    TODO: select mod from a list of tuples with (mean,variance)
  '''
  num_mods = int(num_mods)
  if type(data) != type(None):
    shape = np.shape(data)  
  c = np.int(shape[0]/num_mods)
  np.random.set_state = 1440*shape[0]

  if type(data) != type(None):
    if random_mods:
      if len(shape) > 1:
        mix = lambda data: (data + np.array([np.random.normal(mean_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),var_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),(c,shape[1:]) ) for i in range(num_mods)]).flatten() )
      else:
        mix = lambda data: (data + np.array([np.random.normal(mean_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),var_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),c) for i in range(num_mods)]).flatten() )    
    else:
      if len(shape) > 1:
        mix = lambda data: (data + np.array([np.random.normal(mean_base*(i+1),var_base*(i+1),(c,shape[1:]) ) for i in range(num_mods)]).flatten() )
      else:
        mix = lambda data: (data + np.array([np.random.normal(mean_base*(i+1),var_base*(i+1),c) for i in range(num_mods)]).flatten() )      
  else:
    if random_mods:
      if len(shape) > 1:
        mix = lambda data: (np.array([np.random.normal(mean_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),var_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),(c,shape[1:]) ) for i in range(num_mods)]).flatten() )
      else:
        mix = lambda data: (np.array([np.random.normal(mean_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),var_base*np.mod(np.random.randint((i+1),high=1234*i+4),num_mods),c) for i in range(num_mods)]).flatten() )    
    else:
      if len(shape) > 1:
        mix = lambda data: (np.array([np.random.normal(mean_base*(i+1),var_base*(i+1),(c,shape[1:]) ) for i in range(num_mods)]).flatten() )
      else:
        mix = lambda data: (np.array([np.random.normal(mean_base*(i+1),var_base*(i+1),c) for i in range(num_mods)]).flatten() )      
  return mix(data)

"""#Architecture Basics

##  Gated CNN
  Gated CNN is a doubled CNN in whom one of the convoluted signals does the role of opening/closing the network, giving an **Attention Mechanism** to the convolution, for being activated by a sigmoid.
  It gives non-vanishing gradient, since the multiplication rule for the derivative applies, also, applies gradient to the linear convoluted part.

### GCNN 1D Time Series
  In time series version, since the desirable learning is based on **past** events, or you cannot uphold the assumption that you have acess to future data, have to make sure that the convolution is **causal**, that is
\begin{align}
y_{n}= a_{i}x_{n-i}=a_{n-j}x_{j}
\end{align} 
  Giving at last, if the filter has length k, k-1 zero padding to the input x.
"""

ts  = sample_gaussmix(shape=(10000,), mean_base=0.5, var_base=0.15, random_mods=True, num_mods=50)

batch = 2
data = ts
channels = 3

l = np.shape(ts)
data_tensor = np.ndarray( shape=(batch,l[0],channels) )

'''
Conv1D
__init__(
    filters(channels_out),
    kernel_size(filters),
    strides=1,
    padding='valid',
    data_format='channels_last',
    dilation_rate=1,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
'''

graph = tf.Graph()
with graph.as_default():
  signal_in = tf.placeholder(tf.float32,(None,l[0],3), name='signal_in')
  ####
  ## Keras convolutions. Classes, so dont behave like functions but outputs Tensor, use its functions to query the variables filters and bias
  conv_linear = tf.keras.layers.Conv1D( 4, 8000, padding='causal', name='conv_linear', use_bias=True)(signal_in)
  conv_gate = tf.sigmoid(tf.keras.layers.Conv1D( 4, 8000, padding='causal', name='conv', use_bias=True )(signal_in),name='conv_sigmoid')
  gated_convolutions = tf.multiply(conv_linear,conv_gate,name='gated_convolutions')
  #probability = tf.nn.softmax(gated_convolutions, axis=None, name='probability')
  #minimize_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(-loss)
  
  for op in graph.get_operations():
    print(op.name)
  graph_def = graph.as_graph_def()
  show_graph(graph_def)

####
## Run the constructed graph
#
with graph.as_default():
  session = tf.Session()

  feed_dict = {
      signal_in: data_tensor      
  }
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  session.run(tf.global_variables_initializer())
  lossval = np.zeros(140)
  # Perform  gradient descent steps    
  for step in range(2):
    
    #loss_value = session.run(loss, feed_dict)    
    #lossval[step] = loss_value

    #if step % 1 == 0:
      #print("Step:", step, " Loss:", loss_value)
      #if step % 5 == 0 and step != 0:
        #loss_diff = np.diff(lossval[np.nonzero(lossval)])
        #print("Mean Loss Growth ",np.mean(loss_diff) )    
      
    gated_convs_value = session.run(graph.get_tensor_by_name('gated_convolutions:0'), feed_dict = feed_dict, options=options, run_metadata=run_metadata)
    print('Shape gated_convs ',np.shape(gated_convs_value))

"""### GCNN 1D Residuals"""

graphr = tf.Graph()
with graphr.as_default():
  signal_in = tf.placeholder(tf.float32,(None,l[0],4), name='signal_in')
  ####
  ## Keras convolutions. Classes, so dont behave like functions but outputs Tensor, use its functions to query the variables filters and bias
  conv_linear = tf.keras.layers.Conv1D( 4, 8000, padding='causal', name='conv_linear', use_bias=True)(signal_in)
  conv_gate = tf.sigmoid(tf.keras.layers.Conv1D( 4, 8000, padding='causal', name='conv', use_bias=True )(signal_in),name='conv_sigmoid')
  gated_convolutions = tf.multiply(conv_linear,conv_gate,name='gated_convolutions')
  residual = tf.add(gated_convolutions,signal_in,name='residual')
  
  for op in graphr.get_operations():
    print(op.name)
  graph_def = graphr.as_graph_def()
  show_graph(graph_def)

####
## Run the constructed graph
#
with graphr.as_default():
  session = tf.Session()

  feed_dict = {
      signal_in: data_tensor      
  }
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  session.run(tf.global_variables_initializer())
  lossval = np.zeros(140)
  # Perform  gradient descent steps    
  for step in range(2):
    
    #loss_value = session.run(loss, feed_dict)    
    #lossval[step] = loss_value

    #if step % 1 == 0:
      #print("Step:", step, " Loss:", loss_value)
      #if step % 5 == 0 and step != 0:
        #loss_diff = np.diff(lossval[np.nonzero(lossval)])
        #print("Mean Loss Growth ",np.mean(loss_diff) )    
      
    gated_convs_value = session.run(graphr.get_tensor_by_name('residual:0'), feed_dict = feed_dict, options=options, run_metadata=run_metadata)
    print('Shape gated_convs ',np.shape(gated_convs_value))
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

#Device, filter dimension calculator, arch block helpers

_conv_dim = lambda x,xx,dilation,stride : np.ceil((x-(xx-1)*dilation)/stride)

_ff = lambda ord,loss : 10.0*ord if loss//ord < 1.0/ord else ord
_fff = lambda ord,loss : ord/10.0 if loss//ord > 1.0/(ord/10.0) else ord
ffff = lambda ord,loss: _fff(ord,loss) if _ff(ord,loss) == ord else _ff(ord,loss)

def desambiguatename(opnamescope, scope):   
  '''
   Returns a new name according to the givem op namescope and the scope to embed the op namescope
  ''' 
  istag = scope.find('_')
  first = opnamescope.find('/')  
  appendname = scope.find('//')
  if scope == '' or (appendname != -1 and scope.split('//')[0] == orgname.split('/')[0]):  
    new_name = orgname
  elif istag != -1:    
    if appendname != -1:
      names = orgname.split('/')
      if names[0] == scope[:istag]:
        new_name = names[0]+scope[istag:appendname]+orgname[first:]
      elif scope[:istag] != "":
        new_name = scope[:istag]+orgname[first:] 
      else:
        new_name = names[0]+scope+orgname[first:] 
    else:
      names = orgname.split('/')
      if names[0] == scope[:istag]:
        new_name = names[0]+scope[istag:]+orgname[first:]
      elif scope[:istag] != "":
        new_name = scope[:istag]+orgname[first:] 
      else:
        new_name = names[0]+scope+orgname[first:]         
  elif appendname != -1 and scope.split('//')[0] != orgname.split('/')[0]:
    new_name = scope[:appendname]+'/'+ orgname
  else:
    new_name = scope + orgname[orgname.find('/'):]   

  return new_name    

def get_listeners(graph, scope = 'signal_in/'):
  '''
   Returns the name of the inputs from scopes that has the given scope as input.
  '''
  inputmap = {}
  for node in graph.as_graph_def().node:    
    for inp in node.input:        
      if inp.startswith(scope) and (node.name.startswith(scope) == False):        
        if inputmap.get(inp) == None:
          inputmap[inp] = [node.name]
        else:
          inputmap[inp].append(node.name)
  if len(inputmap.items()) == 0:
    raise NameError('{} is not input of any Arch-Block'.format(scope) )
    
  return inputmap

def get_op_fromscope(graph,scope='signal_in/',opname = 'init', **kwargs):
  '''
   Return operation inside a namescope, independently of how many name scope the op is nested in.   
  ''' 
  
  try:
    op = graph.get_operation_by_name(scope+opname)
    return op
  except:
    notfound = True
    
  if notfound:
    op = []
    for node in graph.as_graph_def().node:    
      if (node.name.startswith(scope) == True) and node.name.split('/')[-1] == opname: 
        op.append(node.name)
    if len(op) != 0:
      return op   
    else:
      raise NameError('Operation not found')

def calculatefilter(initialdim, filterlist):

  a = initialdim  
  b = 0 
  c = 0
  for filter in filterlist:
    a = _conv_dim(a,filter,1,1)
    if a < 0:
      b = a
      c = 1
      break
  if c == 1:
    a = b
  return np.int32(a)

def get_tensor_list_from_scopes(graph, tensor_scope_name):
    
    names = []
    for i in range(len(tensor_scope_name)):
      names.append(graph.get_tensor_by_name(tensor_scope_name[i]+':0'))
    if len(names) > 1:      
      return tf.keras.layers.concatenate( names, axis = 1)
    else:
      return names[0]

def find_softmax(graph, **kwargs):

  names = []    
  for op in graph.get_operations():     
    name = op.name   
    a = name.find('softmax')                
    if a != -1:
      b = name.split('/')
      if b[-2].find('softmax') != -1 and b[-1] == 'transpose_1':
        names.append(op.name)
  
  return names

class dev_selector:

  def __init__(self,**kwargs):
    self.arg1 = kwargs['arg1']
    
  def __call__(self,op):
    if op.find('gcnn2d') != -1:
      return '/device:CPU:0'
    else:
      return '/device:CPU:0'

  def _conv2D_on_gpu(op):
    if op.type == "Conv2D":
      return "/device:GPU:*"
    else:
      return "/device:CPU:*"

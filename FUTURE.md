# Predicted changes that would suite very well the lib.

##FUTURE: use arch block class with enums specifing the blocks
##   Auto compile directives from json and query parameters,
## return requisition or error if parameter not given or is not from appropriated shape and type. 

## Build architecture from block, need changes, possibly in Tensorflow, to be possible to work

Build block from name in the top level of the blocks, this is a previous version that take names inside namescope dict and build new block

```python

  def from_block(self,archblockname,tag = ""):
        
    with self.graph.as_default():
      sess = tf.Session(graph=self.graph)

      new_saver = tf.train.import_meta_graph(os.getcwd()+'/'+self.arch_blocks[archblockname]+'.meta', import_scope = archblockname+tag)
      new_saver.restore(sess, os.getcwd()+'/'+self.arch_blocks[archblockname])
      
      self.signal_in.append(self.graph.get_tensor_by_name(archblockname+tag+'/signal_in:0'))
```

Similar to the previous code, you could define an architecture block and save it, for future versions.

```python
  def define_block(self,blockname):
   
    graph = tf.Graph()
    with graph.as_default():
      sess = tf.Session(graph=graph)
      
      new_saver = tf.train.import_meta_graph(os.getcwd()+'/'+self.arch_blocks[blockname]+'.meta', import_scope = blockname)
      new_saver.restore(sess, os.getcwd()+'/'+self.arch_blocks[blockname])

      self.graph = graph
      self.signal_in = []
      self.signal_in.append(self.graph.get_tensor_by_name(blockname+'/signal_in:0'))
``` 

## Data pipeline for setting up hookers with slices of batch tensors

```python
'''
 placeholder p/ data e label p train e test -> Dataset.from_tensor_slices(data,label) -> configurar batch_size e numero de epocas ->data.Iterator.from_structure(out type, out shape) apenas um é reusavel para qualquer tensor de mesma estrutura -> data=iterator.get_next() -> inicializadores de iteradores -> rede com accuracia e global step no minimize recebe data do iterator ver no apidocs next() e setar sumários -> setar hooks de sumários e checkpoints -> Sessão Monitorada para treino e para teste
'''
self.training_data = (train_images, train_labels.astype(np.int64))  # self.get_one_hot(train_labels, 10))
self.testing_data = (test_images, test_labels.astype(np.int64))  # self.get_one_hot(test_labels, 10))
return tf.data.Dataset.from_tensor_slices(self.testing_data)
iterator = tf.data.Iterator.from_structure(train_ds.output_types,train_ds.output_shapes)
data = iterator.get_next()
``` 

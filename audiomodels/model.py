# -*- coding: utf-8 -*-

from .spect import *
from .cgraphviz import *
from .tfhelper import *
from .dataloader import *
from .ckpt_hookers import *
from .domain import *
from .hyperparamtuning import  *
from .util import *
from .builder import *

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
            Util.set_with_assert(self, 'config', config, str)
            print('config', self.config)
            # assert it is json config file
            if self.config.find('.json') is not -1:
                self.read_from_json()
                print('0.')
                self.configure()
            else:
                raise('config keyword arg must be a string with full path to a valid .json file')
        else:
            if Util.assert_instance(data, BaseDataSampler):
                self.data=data
            else:
                raise ValueError("data must be an instance of BaseDataSampler.")
            if Util.assert_instance( arch_search, Hyperparameter):
                self.arch_search=arch_search
            else:
                raise ValueError("arch_search must be a instance of Hyperparameter")

        if verbose:
            self.verbose=True

    def print_namescopes(self, mode='std_out'):
        '''
        printar buider.namescopes
        '''
        if mode == 'std_out':
            print(self.builder.namescopes.items())

    def configure(self):

        if self.data_loader:
            # Data Loader initializators
            if issubclass(self.data_loader, Sass):
                self.data_loader=self.data_loader(
                    self.data_path,
                    **self.data
                )
                print('1.a')
                del self.data
        if self.arch_search:
            if issubclass(self.arch_search, GCNNMaxPooling):
                self.arch_search=self.arch_search(
                    self.data_domain,
                    **self.framing
                )
                print('1.b')
                del self.framing

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

            Returned as attribute of the Model

           -data_domain: list of lists containing the bounds of the
            dimensions, transformed to tuple on purpose of hashing.

            Returned keys of framing dictionary of the Model Instance. Will
            be used as keywordarg of the Hyperparameter

            -number_of_steps: number of frames to be generated by the slicer.
            -frame_selection: mode of selection of the generated frames,
            default is 'fraction', optionally 'all'.
            -frame_fraction: number between 0.0 and 1.0, the fraction of frames
            to be selected in the case of the frame_selection be 'fraction'.
            -name: name of the class to be instantied.

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
        if "data" in config:
            self.data=config['data']
            if "data_path" in self.data:
                self.data_path=self.data.pop('data_path')
            else:
                raise KeyError('data_path key not found.')
            if "num_samples" in self.data:
                pass
            else:
                self.data['num_samples']=10
            if "number_of_batches" in self.data:
                pass
            else:
                self.data['number_of_batches']=4
            if "split_tax" in self.data:
                pass
            else:
                self.data['split_tax']=0.2
            if "name" in self.data:
                #Data-loader types reader
                name=self.data.pop('name')
                if name=='Sass' or\
                    name=='Stattus4AudioSpectrumSampler':
                    self.data_loader = Sass
                    if "freq_size" in self.data:
                        pass
                    else:
                        self.data['freq_size']=200
                    if "time_size" in self.data:
                        pass
                    else:
                        self.data['time_size']=200
                    if "same_data_validation" in self.data:
                        pass
                    else:
                        self.data['same_data_validation']=True
        else:
            raise KeyError('no data field, in json configuration file')
        if "framing" in config:
            self.framing=config['framing']
            if "data_domain" in self.framing:
                if issubclass(self.framing['data_domain'].__class__,list):
                    if issubclass(self.framing['data_domain'][0].__class__,list):
                        self.data_domain=tuple([tuple(a) for a in self.framing.pop('data_domain')])
                    else:
                        raise Exception('dimension bounds are not properly setted')
                else:
                    raise Exception('domain must be a list')
            else:
                raise KeyError('.json does not have data_domain list object')

            if "number_of_steps" in self.framing:
                pass
            else:
                self.framing['number_of_steps']=10
            if "frame_selection" in self.framing:
                pass
            else:
                self.frame_selection['frame_selection']='fraction'
            if "frame_fraction" in self.framing:
                pass
            else:
                self.framing['frame_fraction']=0.15
            if "name" in self.framing:
                name=self.framing.pop('name')
                if name=='EmbeddedSlicer':
                    self.framing['slicer']=EmbeddedSlicer
                    if "frater_slicer" in self.framing:
                        if self.framing['fater_slicer']=='LadderSlicer':
                            self.framing['fater_slicer']=LadderSlicer
                        elif self.framing['fater_slicer']=='NoSliceSlicer':
                            self.framing['fater_slicer']=NoSliceSlicer
                        else:
                            raise ValueError('Given Unknown Fater Slicer')
                    else:
                        self.framing['fater_slicer']=LadderSlicer
                    if "mater_slicer" in self.framing:
                        if self.framing['mater_slicer']=='LadderSlicer':
                            self.framing['mater_slicer']=LadderSlicer
                        elif self.framing['mater_slicer']=='NoSliceSlicer':
                            self.framing['fater_slicer']=NoSliceSlicer
                        else:
                            raise ValueError('Given Unknown Mater Slicer')
                    else:
                        self.framing['mater_slicer']=LadderSlicer
                    if "recursive" in self.framing:
                        pass
                    else:
                        self.framing['recursive']=True
                    if "recursive_depth" in self.framing:
                        pass
                    else:
                        self.framing['recursive_depth']=2
                    if "recursive_generator" in self.framing:
                        pass
                    else:
                        self.framing['recursive_generator']="Fater"
                if name == 'LadderSlicer':
                    self.framing['slicer']=LadderSlicer
                if name == 'NoSliceSlicer':
                    self.framing['slicer']=NoSliceSlicer
        if "arch_search" in config:
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
        pass

    def run(self, feed_dict, op_to_run = '', number_of_runs = 10, mode =
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



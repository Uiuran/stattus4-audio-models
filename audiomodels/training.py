# encoding : utf-8

from src.model import *

# Define Filter-Bank of the CNN 
filterlist = []
filterlista = [3,3,3,3]
filterlist.append(filterlista)
a = calculatefilter(600, filterlista)

# Max pooling

filterlistb = [5,5,5,5]
filterlist.append(filterlistb)
b = calculatefilter(a/3.0, filterlistb)

# Max pooling

filterlistc = [7,7,7,7]
filterlist.append(filterlistc)
c = calculatefilter(b/2., filterlistc)

# Max pooling 

filterlistd = [5,5,5,5]
filterlist.append(filterlistd)
d = calculatefilter(c/2.0, filterlistd)


# Max pooling 

filterliste = [5,4]
filterlist.append(filterliste)
e = calculatefilter(d/2.0, filterliste)

#Model Building

buildgraph = Builder(dtype=tf.float32, datasize=(600,8), num_input = 10, channels=1)
buildgraph.get_directives('gcnn2d')
buildgraph.get_directives('softmax')
#buildgraph.set_archname('frame')
buildgraph.get_directives('reducemean')
buildgraph.get_directives('losscrossentropy')
pooling = [3,2,2,2]

for i in range(10):

    buildgraph.config_block_name(deepness=len(filterlist[0]),numblock=i)
    for j in np.arange(len(filterlist)):

        cout = 64 + j*20
        if j != 0:
            buildgraph.config_block_name(deepness=len(filterlist[j]),numblock=i)

            for z in np.arange(len(filterlist[j])):

                if z == 0 and j == 0:
                    buildgraph.build_graph_module('cnn2d', channels_out = 64,filter_size=(filterlist[j][0],8), isinput=True, lastnamescope=True, verbose=False)

      # z:= layer onde esta o gate dentro de um block de cnn, entre maxpoolings.
      # j:= número do block de cnn, onde esta o layer z de gate
                if z ==  (len(filterlist[j])-1) and (j == 4):
                    buildgraph.build_graph_module('gcnn2d', channels_out = cout,filter_size=(filterlist[j][z],1), isinput=False, lastnamescope=True, verbose=False)
                else:
                    buildgraph.build_graph_module('cnn2d', channels_out = cout,filter_size=(filterlist[j][z],1), isinput=False, lastnamescope=True, verbose=False)
 
        if j < len(filterlist) - 1:
            buildgraph.config_block_name(deepness=-1,numblock=i,pool=j)
            buildgraph.build_graph_module('maxpooling2d', poolsize = (pooling[j],1), isinput=False, lastnamescope=True, verbose=False)
  
    buildgraph.config_block_name(deepness=0,numblock=i)
    buildgraph.build_graph_module('softmax', num_labels=2, isinput=False, lastnamescope=True, show_cgraph=False)

buildgraph.config_block_name(deepness=0,numblock=0)
buildgraph.build_graph_module('reducemean', isinput=False, lastnamescope=True)
buildgraph.config_block_name(deepness=0,numblock=0)
buildgraph.build_graph_module('losscrossentropy', num_labels = 2, isinput=False, lastnamescope=True)

#LOAD DATA
# Stattus4 Data-Location and sample_num train 160016, test 40000
# st4@new-ia:~/bases/full_db/train$ 
# st4@new-ia:~/bases/full_db/test$
#############

dataload = Sass('/bases', num_samples=8, number_of_batches=1, split_tax=.9, freq_size=600, time_size=80, same_data_validation=True)

# Label Vector

ltrain = []
ltest = []

for i in range(14):    
  ltrain.append([0,1])   
  ltrain.append([1,0])    

# LOAD and RE_RUN
EPOCHS = 100
BATCHS = 50
BATCH_SIZE = 100

for j in range(BATCHS):

  dataload = Sass('/home/penalvad/stattus4/dnn_test_data/', num_samples=50, number_of_batches=1, split_tax=.8, freq_size=600, time_size=80, same_data_validation=True)

  graph, feed_dict, minimize_op, loss, acc = restore_model(model_dir_name = './ckpt/model', data_train, ltrain)

  for i in range(EPOCHS):
    lossval,_ = sess.run([loss,minimize_op], feed_dict=feed_dict)
    accval = sess.run([acc], feed_dict=feed_dict)
    print('loss ',lossval)
    print('acc ', accval)
    print(2*'\n')

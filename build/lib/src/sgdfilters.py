# -*- coding: utf-8 -*-

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
import os

# name of files with data
files = [f for f in os.listdir('/content/datawav')]
refs = [f for f in os.listdir('/content/dataref')]
data,fs,enc = audio.wavread('/content/dataref/'+refs[0])
l = np.size(data)

# Numpy tensors to hold data for tensorflow data holders.
# Shape is Batch_Size x Data_Size(dim1*dim1_size,dim2*dim2_size, and so on ..., only dim1 for time series) x Num_Channels (e.g. 3 for images and N for a eeg data batch).
tensor_data = np.ndarray(shape=(1,l,1))
# This tensor will be used as reference for the optimization process. 
tensor_ref = np.ndarray(shape=(5,l,1))

#Load references
for r in range(len(refs)):

  data1,_,_ = audio.wavread('/content/dataref/'+refs[r])
  tensor_ref[r,:,0] = data1.copy()
  
#Load data
for f in range(len(files)):    
  data,_,_ = audio.wavread('/content/datawav/'+files[f])
  tensor_data[f,:,0] = data.copy()

# Comp. Graph
# TODO: encapsulate the comp-graph building/serial format saving.
graph = tf.Graph()
with graph.as_default():

  signal_in = tf.placeholder(tf.float32,(None,l,1), name='signal_in')  
  filter = tf.get_variable('filter', shape=[8000,1,1],initializer=tf.random_normal_initializer(), dtype=tf.float32)
  w = tf.get_variable('w', shape=[5,l],initializer=tf.random_normal_initializer(), dtype=tf.float32)
  signal_ref = tf.placeholder(tf.float32,(None,l,1), name='signal_ref')

  # 1D Convolve  which internally uses 2D reshaped https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
  signal_out = tf.nn.conv1d(signal_in,filter,1,'SAME', name='signal_out')  
  loss = tf.reduce_sum(tf.math.squared_difference(signal_out,tf.tensordot(w,signal_ref,[[0,1],[0,1]], name='ref_contraction'), name = 'squared'), name = 'loss')
  minimize_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

  # Print operations for the graph built
  for op in graph.get_operations():
    print(op.name)

# Create session and run graph many times, until the filter is fitted to minimize loss function
# TODO: encapsulate this
with graph.as_default():
  session = tf.Session()

  feed_dict = {
      signal_in: tensor_data,
      signal_ref:tensor_ref   
  }

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  session.run(tf.global_variables_initializer())
  lossval = np.zeros(140)

  # Perform  gradient descent steps    
  for step in range(140):
    
    loss_value = session.run(loss, feed_dict)    
    lossval[step] = loss_value

    if step % 1 == 0:
      print("Step:", step, " Loss:", loss_value)
      if step % 5 == 0 and step != 0:
        loss_diff = np.diff(lossval[np.nonzero(lossval)])
        print("Mean Loss Growth ",np.mean(loss_diff) )
          
    session.run(minimize_op,
                feed_dict = feed_dict,
                options=options,
                run_metadata=run_metadata)    
    
    
    # Profiling time of operation
    #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    #chrome_trace = fetched_timeline.generate_chrome_trace_format()
    #with open('timeline_0_0_step_%d.json' % step, 'w') as f:
      #f.write(chrome_trace)

# Plot

  signal_out_value = session.run(graph.get_tensor_by_name('signal_out/Squeeze:0'), feed_dict)  
  filter_value = filter.eval(session=session)
  print('Wiener filter ')
  plt.figure()
  plt.plot(filter_value[:,0,0])    
  print('output_filter_SGD')
  plt.figure()
  plt.plot(signal_out_value[0,:,0])  
  print('input_signal')  
  plt.figure()
  plt.plot(tensor_data[0,:,0])

  audio.wavwrite( signal_out_value[0,:,0]/(np.std(signal_out_value[0,:,0],ddof=1.0)+1e-10),'/content/max_0_12072019.wav', fs=fs, enc=enc)
  
  frozen_graph = freeze_session(session, output_names= None)
  tf.train.write_graph(frozen_graph, "model", "./wiener.pb", as_text=False)

# Save reference optimized filtered signal and an one db louder version of the original audio
  audio.wavwrite(signal_out_value[0,:,0],'/content/filtered.wav', fs=fs, enc=enc)
  audio.wavwrite( 10.0*tensor_data[0,:,0],'/content/input.wav', fs=fs, enc=enc)


# Below functions are done in order to decimate time-series acording to its Spectrogram and Nyquist Theorem to use smaller time series with full information.

def downsampling(data,num):
  '''
   Downsampling por num(int) feito como se fosse no C
  '''
  d = np.copy(data)
  j=0
  for i in range(len(d)):

    if i%int(num) == 0:
      j +=1
    else:
      d[i] = 0.0
  datadown = np.zeros(j)
  j=0
  for i in range(len(d)):
  
    if d[i] != 0.0: 
      #print(d[i])
      datadown[j] = d[i]
      j += 1  

  return datadown

def amostragem(data, fs, plot = False):
  '''
   Amostragem de downsampling de uma serie temporal segundo Nyquist-Shannon. Alem disso o numero maximo
   de frequencias com o spectro calculado por ciclo deve ser igual ao numero de pontos N/2 (principio de incerteza).

   
   argumentos:
   
   - data: serie temporal
   - fs: frequência de amostragem do encode original
   - plot: opcional, plota o espectrograma com parametros padrões para obter máxima frequência maior que a largura de banda.
  '''
  
  f,t,Sxx = signal.spectrogram(data,fs,nfft=fs/4,nperseg=fs/5,noverlap=fs/10,scaling='spectrum',mode='magnitude')
  freq = np.array([i for i in zip(t,Sxx.T) if np.shape(f[i[1]>np.std(i[1])])[0] != 0])
  
  if plot:
    plt.figure()
    plt.pcolormesh(t, f, np.log(Sxx+1e-13) )
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

  return [[ i[0], np.ceil(np.float(fs)/(2.0*np.max(f[i[1]>np.std(i[1])]) )) ] for i in freq]

####
## Spectrogram + Downsampling according to Nyquist-Shannon.
#

sampa = np.min(np.array(amostragem(measure,fs, plot=True))[:,1])
sampb = np.min(np.array(amostragem(airnoise,fs1, plot=True))[:,1])
fsa = (fs/int(sampa))
fsb=(fs1/int(sampb))

a = downsampling(measure,sampa )
b = downsampling(airnoise,sampb )

f,t,Sxx = signal.spectrogram(a,fsa,nfft=fsa/4,nperseg=fsa/5,noverlap=fsa/10,scaling='spectrum',mode='magnitude')
plt.figure()
plt.pcolormesh(t, f, np.log(Sxx+1e-13) )
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

f,t,Sxx = signal.spectrogram(b,fsb,nfft=fsb/4,nperseg=fsb/5,noverlap=fsb/10,scaling='spectrum',mode='magnitude')
plt.figure()
plt.pcolormesh(t, f, np.log(Sxx+1e-13))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


plt.figure()
plt.plot(measure,label='original')
plt.figure()
plt.plot(a,label='downsampled')
plt.figure()
plt.plot(airnoise,label='original')
plt.figure()
plt.plot(b,label='downsampled')


audio.wavwrite(a,'/content/a.wav',fs=fsa,enc=enc)
audio.wavwrite(b,'/content/b.wav',fs=fsb,enc=enc)

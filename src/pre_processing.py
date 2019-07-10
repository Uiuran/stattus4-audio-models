# Pre-Processing Functions

import librosa
from librosa.feature import melspectrogram
import numpy as np
import os

# Raw data to Chunk Mel-Spectogram 
# Read data from the hard-drive '/home/stattus4dpenalva/stattus4/wav_data_2018/'
def framewise_mel(datapath, frame_size, batchsize, n_mels, fmax=8000):
  '''
    Input:
    - n_mels: is the size of MEL spectrogram filter outputed as feature of the frame.
    - frame_size: the size of the frame in the time domain. In 16khz you have 16 samples/ms.
    - batch_size: the number of files to read from datapath.
    
    Returns batch_size x number_frames x n_mels_features x time_steps , label_list       
  '''
  spect_list = []
  nomes = [f for f in os.listdir(datapath)]
  label_list = []
  batch_size = len(nomes)
  batch_size = batch_size[0:batchsize]
  batch_mel = []

  ## Frame Parameters
  frame_size = frame_size # 516 is 32 ms for 16khz, choosing 32ms frames like the reference paper

  #framed_data = np.ndarray(shape=(batch_size, chunk, )
  for i in range(batch_size):
  
    data,fs = librosa.load(datapath+nomes[i],sr=None)
    num_frames = len(data)//(frame_size/2)
    label_list.append(nomes[i][0:2])
      
    for j in range(num_frames-2): 
      chunk = data[j*(frame_size/2):j*(frame_size/2)+frame_size]
      spect_list.append( melspectrogram(y=chunk, sr=fs, n_mels=n_mels, fmax=fmax) )
    
    spect_list = np.array(spect_list) # use np.concatenate(spect_list,axis=1) to output the full MEL spect 
    batch_mel.append(spect_list)
    
  return np.array(batch_mel),np.array(label_list)

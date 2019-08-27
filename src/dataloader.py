#encode : utf-8

from template import BaseDataSampler, NotEnoughDataError
import os
import numpy as np
from spect import *
import librosa
from random import shuffle

class Stattus4AudioSpectrumSampler(BaseDataSampler):
  '''
     Data-loader for binary classified (see-below) mono audio (.wav).
    The loader automaticly transforms the data in a Spectrogram with given or standard size using windowed fft.
    It raises an exception if it cant find enough data of one of the labels.

    The labels are sv (short for without leaky 'sem vazamento' in portuguese) and cv (with leaky 'com vazamento' in portuguese). They are given as the first 2 chars from the audio file name.

    Note: this is a class equilibrated data sampler, it's not implemented for audio data that has much more of one label than the other.
  '''
  
# Doing divide by sample label and batch_size, do generator with yield
  def __init__(self, data_dir, num_samples = 50, number_of_batches = 10, split_tax = 0.2, freq_size = 600, time_size = 50, same_data_validation = True):
    '''
      Given data_dir (directory of data) it injects an amount of data of number_of_batches with 2*num_samples for each batch into memory (num_samples for each labeled data).
      Data points are a tuple with unique_id in position 0, label in position 1 and spectrogram in position 2. The unique_id and the label are extracted from audio filename loaded.
    '''         
    super(Stattus4AudioSpectrumSampler,self).__init__(data_dir)  

    self.split_tax = split_tax
    self.nbatches = number_of_batches
    self.sampled_batches = [0,0]
    self.num_samples = num_samples
    self.freq_size = freq_size
    self.time_size = time_size

    if self.split_tax > 0.5:
      self.num_of_training_samples = (self.num_samples*self.split_tax//1)
      self.num_of_test_samples = (self.num_samples*(1.-self.split_tax)//1)  
    else:
      self.num_of_training_samples = (self.num_samples*(1.0-self.split_tax)//1)
      self.num_of_test_samples = (self.num_samples*self.split_tax//1) 

    self.data_dir = data_dir
    self.nomes = [f for f in os.listdir(data_dir)]
    self.data_list_cv = []
    self.data_list_sv = []
    # Versao Final estrutura de dados ([], [], np.ndarray())

    c = 0
    b = 0 
    m = 0
    batchcount = self.nbatches

    self.data_train = []
    self.data_test = []
    self.data_valid = []
    self.datavalidsame = same_data_validation    

    for i in range(len(self.nomes)):
    
      if (self.nomes[i].find('sv') != -1 or self.nomes[i].find('SV') != -1) and c < self.num_samples:

        label = self.nomes[i][0:2]    
        unique_id = self.nomes[i][2:-4] 
        data,fs = librosa.load(self.data_dir+self.nomes[i],sr=None)
        spec = subbed_spect(data,fs,plot=False)
        m = np.shape(spec[0])
    
        if m[0] > self.freq_size and m[1] > self.time_size:
  
          self.data_list_sv.append( (unique_id, label, spec[0][:self.freq_size,:self.time_size]) )
          c += 1          

      if (self.nomes[i].find('cv') != -1 or self.nomes[i].find('CV') != -1 )and b < self.num_samples:

        label = self.nomes[i][0:2]    
        unique_id = self.nomes[i][2:-4]
        data,fs = librosa.load(self.data_dir+self.nomes[i],sr=None)
        spec = subbed_spect(data,fs,plot=False)
        m = np.shape(spec[0])
    
        if m[0] > self.freq_size and m[1] > self.time_size:

          self.data_list_cv.append( (unique_id, label, spec[0][:self.freq_size,:self.time_size]) )
          b += 1   
        
      if b == self.num_samples and c == self.num_samples and batchcount > 0:
        batchcount -= 1
        b = 0
        c = 0
      elif batchcount < 0:
        break

    if batchcount > 0:
      raise NotEnoughDataError('Not enough Data to Sample the Required Training Protocol', b, c, self.nbatches-batchcount)

  def training(self):
    '''
      Training point is returned as a list of lists, being each list a label CV data point(tuple) in position 0 and a label SV data point(also tuple) in position. it is sampled according to the split_tax given in the sampler initializator.
    '''
    
    cv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples) )
    sv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples))
    c = 0
    self.data_train = []                      
      
    while c < self.num_of_training_samples:
          
      datacv = self.data_list_cv[cv]
      cv += 1
      datasv = self.data_list_sv[sv]    
      sv += 1 

      self.data_train.append( [datacv,datasv] )
      c += 1
    
    self.sampled_batches[0] +=  1

    return self.data_train
 
  def testing(self):
    '''
      Testing point is returned as a list of lists, being each list a label CV data point(tuple) in position 0 and a label SV data point(also tuple) in position. it is sampled according to the split_tax given in the sampler initializator.
    '''    
    cv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples))
    sv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples) )
    c = 0
    self.data_test = []                      
      
    while c < self.num_of_test_samples:
          
      datacv = self.data_list_cv[cv]
      cv += 1
      datasv = self.data_list_sv[sv]    
      sv += 1 

      self.data_test.append( [datacv,datasv] )
      c += 1
    
    self.sampled_batches[1] +=  1

    return self.data_test

  def validation(self):
    raise NotImplementedError

Sass = Stattus4AudioSpectrumSampler

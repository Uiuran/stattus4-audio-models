# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg
correlate = signal.correlate
    
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

def subbed_timeseries(measure,fs):
  '''
   Return Sxx (log squared amplitude of stft spectra), f (frequencies), t (time steps)
  '''
  sampa = np.min(np.array(amostragem(np.array(measure),fs, plot=False))[:,1])  
  fsa = (fs/int(sampa))
    
  a = downsampling(np.array(measure),sampa )

  return a,fsa

def subbed_spect(measure,fs,plot=True):
  '''
   Return Sxx (log squared amplitude of stft spectra), f (frequencies), t (time steps)
  '''
  sampa = np.min(np.array(amostragem(np.array(measure),fs, plot=plot))[:,1])  
  fsa = (fs/int(sampa))
    
  a = downsampling(np.array(measure),sampa )
  f,t,Sxx = signal.spectrogram(a,fsa,nfft=fsa/4,nperseg=fsa/5,noverlap=fsa/10,scaling='spectrum',mode='magnitude')
  #Cut zero variance part of spectra(useless recording)   
  #Sxx = Sxx[:,np.var(np.log(Sxx+1e-13),axis=0) > 0.5]
  #l = len(Sxx[0,:])
  #t = t[:l] 
  
  if plot:
    plt.figure()
    plt.pcolormesh(t, f, np.log(Sxx+1e-13))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
  return np.abs(np.log(Sxx+1e-13)),f,t


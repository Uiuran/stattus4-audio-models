#encode : utf-8

from errors import NotEnoughDataError, BatchLimitException
from template import BaseDataSampler
from spect import *
from util import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import librosa
from random import shuffle

class Stattus4AudioSpectrumSampler(BaseDataSampler):
    '''
     Data-loader for binary classified (see-below) mono audio (.wav).
     The loader automaticly transforms the data in a Spectrogram (If it was not a Spectrogram yet) with given or standard size using windowed fft.
     It raises an exception if it cant find enough data of one of the labels.

     The labels are sv (short for without leaky 'sem vazamento' in portuguese) and cv (with leaky 'com vazamento' in portuguese). They are given as the first 2 chars from the audio file name.

     Note: this is a class equilibrated data sampler, it's not implemented for audio data that has much more of one label than the other.
    '''

# Doing divide by sample label and batch_size, do generator with yield
    def __init__(
        self,
        data_dir,
        num_samples=None,
        number_of_batches=None,
        split_tax=None,
        freq_size=None,
        time_size=None,
        same_data_validation=None,
        test_dir=None,
        **kwargs
    ):
        '''
         Given data_dir (directory of data) it injects an amount of data of number_of_batches with 2*num_samples for each batch into memory (num_samples for each labeled data).
         Data points are a tuple with unique_id in position 0, label in position 1 and spectrogram in position 2. The unique_id and the label are extracted from audio filename loaded.
        '''
        BaseDataSampler.__init__(self, data_dir,**kwargs)
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
        self.test_dir = test_dir
        self.nomes = [f for f in os.listdir(data_dir)]

        find_image_ext = np.array([self.nomes[0].find('.png'), self.nomes[0].find('.jpg'), self.nomes[0].find('.jpeg')])
        if (find_image_ext < 0).all():
            self.is_image = False
        else:
            self.is_image = True

        self.data_list_cv = []
        self.data_list_sv = []

        self.data_train = []
        self.data_test = []
        self.data_valid = []
        self.datavalidsame = same_data_validation
        if not self.on_the_fly:
            self._read_data()
        else:
            #Implement counters of data read on the fly
            self._nomes=copy.copy(self.nomes)

    def info(self, mode='verbose'):
        if mode=='verbose':
            print(10*'=')
            print('Split Tax', self.split_tax)
            print('Number of Batches', self.nbatches)
            print('Sampled Batches', self.sampled_batches)
            print('Number of Samples per Batch', self.num_samples)
            print('Frequency Size', self.freq_size)
            print('Time Size', self.time_size)
            print(10*'=')

    def training(self):
        '''
         Training-batch is returned as a list of lists, being each list a label
         CV data point(tuple) in position 0 and a label SV data point(also
         tuple) in position. It is sampled according to the split_tax given in
         the sampler initializator.

         Each Training-batch has num_samples*split_tax number of class
         equalized data-points.
        '''

        if self.sampled_batches[0] == self.nbatches:
            raise BatchLimitException('Not possible to sample more than'+str(self.nbatches)+' training-batches.')

        cv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples))
        sv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples))
        c = 0
        self.data_train = []

        while c < self.num_of_training_samples:
            if self.on_the_fly:
                self._read_data_on_the_fly()

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
         Testing batch is returned as a list of lists, being each list a label CV data point(tuple) in position 0 and a label SV data point(also tuple) in position. it is sampled according to the split_tax given in the sampler initializator.
        '''
        if self.sampled_batches[1] == self.nbatches:
            raise BatchLimitException('Not possible to sample more than'+str(self.nbatches)+' testing-batches.')

        cv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples))
        sv = int(self.sampled_batches[0]*(self.num_of_training_samples) + self.sampled_batches[1]*(self.num_of_test_samples) )
        c = 0
        self.data_test = []

        while c < self.num_of_test_samples:
            if self.on_the_fly:
                self._read_data_on_the_fly()

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

    def _read_data(self):

        c = 0
        b = 0
        m = 0

        batchcount = self.nbatches

        for i in range(len(self.nomes)):

            if (self.nomes[i].find('sv') != -1 or self.nomes[i].find('SV') != -1) and c < self.num_samples:

                label = self.nomes[i][0:2]
                unique_id = self.nomes[i][2:-4]

                if self.is_image:

                    img = load_img(self.data_dir+self.nomes[i], grayscale = True)  # this is a PILLOW image
                    img = img_to_array(img)
                    img /= np.max(img) + 1e-8
                    m = np.shape(img)

                    if m[0] >= self.freq_size and m[1] >= self.time_size:
                        dat = np.ndarray(shape=(self.freq_size,self.time_size))
                        dat = img[:self.freq_size,:self.time_size, 0]
                        c += 1
                        self.data_list_sv.append( (unique_id, label, dat) )
                else:

                    data,fs = librosa.load(self.data_dir+self.nomes[i],sr=None)
                    spec = subbed_spect(data, fs, downsample=False)
                    m = np.shape(spec[0])

                    if m[0] >= self.freq_size and m[1] >= self.time_size:
                        dat = np.ndarray(shape=(self.freq_size,self.time_size))
                        dat[:,:] = spec[0][:self.freq_size,:self.time_size]
                        #dat.tolist()
                        c += 1
                        self.data_list_sv.append( (unique_id, label, dat) )

            if (self.nomes[i].find('cv') != -1 or self.nomes[i].find('CV') != -1 )and b < self.num_samples:

                label = self.nomes[i][0:2]
                unique_id = self.nomes[i][2:-4]

                if self.is_image:

                    img = load_img(self.data_dir+self.nomes[i], grayscale = True)  # this is a PILLOW image
                    img = img_to_array(img) #numpy array
                    img /= np.max(img) + 1e-8
                    m = np.shape(img)

                    if m[0] >= self.freq_size and m[1] >= self.time_size:
                        dat = img[:self.freq_size,:self.time_size,0]
                        b += 1
                        self.data_list_cv.append( (unique_id, label, dat) )
                else:

                    data,fs = librosa.load(self.data_dir+self.nomes[i],sr=None)
                    spec = subbed_spect(data,fs, downsample=False)
                    m = np.shape(spec[0])

                    if m[0] > self.freq_size and m[1] > self.time_size:
                        dat = np.ndarray(shape=(self.freq_size,self.time_size))
                        dat[:,:] = spec[0][:self.freq_size,:self.time_size]
                        #dat.tolist()
                        b += 1
                        self.data_list_cv.append( (unique_id, label, dat) )

            if b == self.num_samples and c == self.num_samples and batchcount > 0:
                batchcount -= 1
                b = 0
                c = 0
            elif batchcount < 0:
                break

        if batchcount > 0:
            raise NotEnoughDataError('Not enough Data to Sample the Required Training Protocol', b, c, self.nbatches-batchcount)

    def _read_data_on_the_fly(self):

        c = 0
        b = 0
        m = 0
        i=0
        lname=len(self._nomes)
        while c+b<self.num_samples:

            if (self._nomes[i].find('sv') != -1 or self._nomes[i].find('SV') != -1) and c < self.num_samples:

                
                label = self._nomes[i][0:2]
                unique_id = self._nomes[i][2:-4]

                if self.is_image:

                    img = load_img(self.data_dir+self._nomes[i], grayscale = True)  # this is a PILLOW image
                    img = img_to_array(img)
                    img /= np.max(img) + 1e-8
                    m = np.shape(img)

                    if m[0] >= self.freq_size and m[1] >= self.time_size:
                        dat = np.ndarray(shape=(self.freq_size,self.time_size))
                        dat = img[:self.freq_size,:self.time_size, 0]
                        c += 1
                        self.data_list_sv.append( (unique_id, label, dat) )
                        self._nomes.pop(i)
                    else:
                        i += 1
                else:

                    data,fs = librosa.load(self.data_dir+self._nomes[i],sr=None)
                    spec = subbed_spect(data, fs, downsample=False)
                    m = np.shape(spec[0])

                    if m[0] >= self.freq_size and m[1] >= self.time_size:
                        dat = np.ndarray(shape=(self.freq_size,self.time_size))
                        dat[:,:] = spec[0][:self.freq_size,:self.time_size]
                        #dat.tolist()
                        c += 1
                        self.data_list_sv.append( (unique_id, label, dat) )
                        self._nomes.pop(i)
                    else:
                        i += 1

            if (self._nomes[i].find('cv') != -1 or self._nomes[i].find('CV') != -1 )and b < self.num_samples:

                label = self._nomes[i][0:2]
                unique_id = self._nomes[i][2:-4]

                if self.is_image:

                    img = load_img(self.data_dir+self._nomes[i], grayscale = True)  # this is a PILLOW image
                    img = img_to_array(img) #numpy array
                    img /= np.max(img) + 1e-8
                    m = np.shape(img)

                    if m[0] >= self.freq_size and m[1] >= self.time_size:
                        dat = img[:self.freq_size,:self.time_size,0]
                        b += 1
                        self.data_list_cv.append( (unique_id, label, dat) )
                        self._nomes.pop(i)
                    else:
                        i += 1
                else:

                    data,fs = librosa.load(self.data_dir+self._nomes[i],sr=None)
                    spec = subbed_spect(data,fs, downsample=False)
                    m = np.shape(spec[0])

                    if m[0] > self.freq_size and m[1] > self.time_size:
                        dat = np.ndarray(shape=(self.freq_size,self.time_size))
                        dat[:,:] = spec[0][:self.freq_size,:self.time_size]
                        #dat.tolist()
                        b += 1
                        self.data_list_cv.append( (unique_id, label, dat) )
                        self._nomes.pop(i)
                    else:
                        i += 1

            if lname-1<i:
                raise NotEnoughDataError('Not enough data from the given sizes',b,c,self.sampled_batches)

Sass = Stattus4AudioSpectrumSampler

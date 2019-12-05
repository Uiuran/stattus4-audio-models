from .dataloader import *
import librosa as audio

path = '/home/penalva/stattus4-audio-models/data/'

def generate_sinusoidal_mock(path, num):
    for i in range(num):
        if i < num//2:
            audio.output.write_wav(path+f"sv_00{i}.wav", np.sin(0.01*i*np.linspace(0,160000,160000)),16000)
        else:
            audio.output.write_wav(path+f"cv_00{i}.wav", np.sin(0.01*i*np.linspace(0,160000,160000)),16000)

def test_load_on_the_fly():
    dataset = Sass(path, num_samples = 10,
                   number_of_batches = 4, split_tax=0.2,
                   freq_size=5, time_size=2,
                   same_data_validation=True, on_the_fly=True, test_dir='')
    return dataset

def test_load_full():

    dataset = Sass(path, num_samples = 10,
                   number_of_batches = 4, split_tax=0.2,
                   freq_size=5, time_size=2,
                   same_data_validation=True, on_the_fly=False, test_dir='')
    return dataset

def test_extra_training_batch():

    dataset = Sass(path, num_samples = 10,
                   number_of_batches = 4, split_tax=0.2,
                   freq_size=5, time_size=2,
                   same_data_validation=True, test_dir='')

    for i in range(dataset.nbatches + 1):
        dtrain = dataset.training()
        print('Number of training points', len(dtrain))
        print('Example Point', dtrain[0])
        dataset.info()

    return dataset

# Corrigir teste de amostragem de teste
def test_testing_batch():

    dataset = Sass(path, num_samples = 10,
                   number_of_batches = 4, split_tax=0.2,
                   freq_size=5, time_size=2,
                   same_data_validation=True, test_dir='')

    for i in range(dataset.nbatches):
        dtest = dataset.testing()
        print('Number of training points', len(dtest))
        print('Example Point', dtest[0])
        dataset.info()

    return dataset

# Teste de amostragem extra de batches for testing
def test_extra_testing_batch():

    dataset = Sass(path, num_samples = 10,
                   number_of_batches = 4, split_tax=0.2,
                   freq_size=6, time_size=2,
                   same_data_validation=True, test_dir='')

    for i in range(dataset.nbatches + 1):
        dtrain = dataset.testing()
        print('Number of training points', len(dtrain))
        print('Example Point', dtrain[0])
        dataset.info()
    return dataset

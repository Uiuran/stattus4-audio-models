from dataloader import *

path = '/home/penalvad/stattus4/benchfull/'

dataset = Sass(path, num_samples = 10,
               number_of_batches = 4, split_tax=0.2,
               freq_size=250, time_size=50,
               same_data_validation=True, test_dir='')

def test_extra_training_batch():
    for i in range(dataset.nbatches + 1):
        dtrain = dataset.training()
        print('Number of training points', len(dtrain))
        print('Example Point', dtrain[0])
        dataset.info()

# Corrigir teste de amostragem de teste
def test_testing_batch():
    for i in range(dataset.nbatches):
        dtest = dataset.testing()
        print('Number of training points', len(dtest))
        print('Example Point', dtest[0])
        dataset.info()

# Teste de amostragem extra de batches for testing
def test_extra_testing_batch():
    for i in range(dataset.nbatches + 1):
        dtrain = dataset.testing()
        print('Number of training points', len(dtrain))
        print('Example Point', dtrain[0])
        dataset.info()

#test_extra_training_batch()
test_testing_batch()
test_extra_testing_batch()

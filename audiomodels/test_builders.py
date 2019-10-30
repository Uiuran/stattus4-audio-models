from hyperparamtuning import GCNNMaxPooling
from dataloader import Sass
from domain import *
from layers import Signal

path='/home/penalva/stattus4-audio-models/data/'
def test_signal():
    hptuning = GCNNMaxPooling(((0,5),(0,2)), slicer= EmbeddedSlicer,
                              fater_slicer = LadderSlicer, mater_slicer = LadderSlicer,
                              number_of_steps=20, frame_selection='fraction',
                              frame_fraction=0.75, recursive_depth = 2,
                              mode='iterator')
    dataset = Sass(path, num_samples = 10,
                   number_of_batches = 4, split_tax=0.2,
                   freq_size=5, time_size=2,
                   same_data_validation=True, test_dir='')

    signal = Signal(dtype=float, number_of_inputs=20, channels=1)(dataset,tuning=hptuning)
    return signal,hptuning

def test_feed_batch():
    signal,_=test_signal()
    signal.feed_batch()
    print('Feed Dictionary')
    print(signal.feed_dict)
    return signal

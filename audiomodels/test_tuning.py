from .hyperparamtuning import *

def test_embeddedslicer():
    hptuning = GCNNMaxPooling(((0,400),(0,600)), slicer= EmbeddedSlicer,
                              fater_slicer = LadderSlicer, mater_slicer = LadderSlicer,
                              number_of_steps=10, frame_selection='fraction',
                              frame_fraction=0.50, recursive_depth = 2,
                              mode='iterator')

    for i in range(len(hptuning.slicer.slices)):
        hptuning.update_architecture()
    print('EmbeddedSlicer GCNN Max Pooling')
    print(hptuning.deep_arch)

    return hptuning

def test_ladderslicer():

    hptuning = GCNNMaxPooling(((0,400),(0,600)), slicer= LadderSlicer,
                              number_of_steps=10, frame_selection='fraction',
                              frame_fraction=0.5)

    print(len(hptuning.slicer.slices))
    for i in range(len(hptuning.slicer.slices)):
        hptuning.update_architecture()
    print('LadderSlicer GCNN Max Pooling')
    print(hptuning.deep_arch)

    return hptuning

def test_nosliceslicer():

    hptuning = GCNNMaxPooling(((0,400),(0,600)), slicer= NoSliceSlicer,
                              number_of_steps=10, frame_selection='fraction',
                              frame_fraction=1.)

    for i in range(len(hptuning.slicer.slices)):
        hptuning.update_architecture()
        print('NoSliceSlicer GCNN Max Pooling')
        print(hptuning.deep_arch)
        print('Final degradation')
        hptuning.eval_metrics(((0,400),(0,600)),hptuning.deep_arch[((0,400),(0,600))],'all')
        print(hptuning.degrad)

        return hptuning

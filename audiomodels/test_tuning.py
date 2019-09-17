from hyperparamtuning import *
import pdb

hptuning = GCNNMaxPooling(((0,40),(0,40)), slicer= EmbeddedSlicer,
                          fater_slicer = LadderSlicer, mater_slicer = LadderSlicer,
                          number_of_steps=10, frame_selection='fraction',
                          frame_fraction=0.10, recursive_depth = 1)


for i in range(len(hptuning.slicer.slices)):
    hptuning.update_architecture()
print('EmbeddedSlicer GCNN Max Pooling')
print(hptuning.deep_arch)

hptuning = GCNNMaxPooling(((0,40),(0,40)), slicer= LadderSlicer,
                          number_of_steps=10, frame_selection='fraction',
                          frame_fraction=0.8)

for i in range(len(hptuning.slicer.slices)):
    hptuning.update_architecture()
print('LadderSlicer GCNN Max Pooling')
print(hptuning.deep_arch)

hptuning = GCNNMaxPooling(((0,40),(0,40)), slicer= NoSliceSlicer,
                          number_of_steps=10, frame_selection='fraction',
                          frame_fraction=1.)

for i in range(len(hptuning.slicer.slices)):
    hptuning.update_architecture()
print('NoSliceSlicer GCNN Max Pooling')
print(hptuning.deep_arch)
print('Final degradation')
hptuning.eval_metrics(((0,40),(0,40)),hptuning.deep_arch[((0,40),(0,40))],'all')
print(hptuning.degrad)

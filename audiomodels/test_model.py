from model import *

model = Model(
            data=Stattus4AudioSpectrumSampler,
            framing=EmbeddedSliceSlicer,
            arch_search=GCNNMaxpooling,
            config='test_model.json',
            verbose=True)

def test_one_batch():
    model.feed_batch(source='model')
    model.run(ckpt='~/stattus4/model_test_ckpt/')
    model.test()

def test_batch_pool():
    model.feed_pool(source='model')
    model.run()
    model.run()
    model.run_all(ckpt='~/stattus4/model_test_ckpt/', strategy='accuracy')
    model.test()

#test_one_batch()
test_batch_pool()

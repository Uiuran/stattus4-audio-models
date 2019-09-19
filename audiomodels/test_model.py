from model import *

model = Model(
            data=Stattus4AudioSpectrumSampler,
            framing=EmbeddedSlicer,
            arch_search=GCNNMaxPooling,
            config='test_model.json',
            verbose=True)

def test_configure():
    model.info()

def test_compile():
    model.print_names()

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

from .model import *

def test_configure():
    model = Model(
        data=Stattus4AudioSpectrumSampler,
        framing=EmbeddedSlicer,
        arch_search=GCNNMaxPooling,
        config='test_model.json',
        verbose=True
    )

    print(model.data_loader.training())
    model.info()
    return model

def test_signal():
    model = Model(
        data=Stattus4AudioSpectrumSampler,
        framing=EmbeddedSlicer,
        arch_search=GCNNMaxPooling,
        config='test_model.json',
        verbose=True
    )
    print(model.signal)
    return model

def test_compile():
    model = Model(
        data=Stattus4AudioSpectrumSampler,
        framing=EmbeddedSlicer,
        arch_search=GCNNMaxPooling,
        config='test_model.json',
        verbose=True
    )
    model.compile()
    model.print_namescopes()
    return model

def test_one_batch():
    model = Model(
        data=Stattus4AudioSpectrumSampler,
        framing=EmbeddedSlicer,
        arch_search=GCNNMaxPooling,
        config='test_model.json',
        verbose=True
    )
    model.signal.feed_batch(source='model')
    model.run(ckpt='~/stattus4/model_test_ckpt/')
    model.test()
    return model

def test_batch_pool():
    model = Model(
        data=Stattus4AudioSpectrumSampler,
        framing=EmbeddedSlicer,
        arch_search=GCNNMaxPooling,
        config='test_model.json',
        verbose=True
    )
    model.feed_pool(source='model')
    model.run()
    model.run()
    model.run_all(ckpt='~/stattus4/model_test_ckpt/', strategy='accuracy')
    model.test()
    return model

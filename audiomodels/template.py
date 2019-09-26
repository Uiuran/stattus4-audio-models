# encode: utf-8

class BaseDataSampler(object):
    """DataSampler which generates a TensorFlow Dataset object from given input.
    """

    def __init__(self, data_dir, on_the_fly=True):
        self.data_dir = data_dir
        self.on_the_fly=on_the_fly

    def training(self):
        raise NotImplementedError()

    def testing(self):
        raise NotImplementedError()

    def validation(self):
        raise NotImplementedError()

    def _read_data(self):
        raise NotImplementedError()

    def _read_data_on_the_fly(self):
        raise NotImplementedError()



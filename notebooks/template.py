# encode: utf-8

class BaseDataSampler(object):
    """DataSampler which generates a TensorFlow Dataset object from given input.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def training(self):
        raise NotImplementedError

    def testing(self):
        raise NotImplementedError

    def validation(self):
        raise NotImplementedError

class NotEnoughDataError(Exception):

    def __init__(self, message, data_cv, data_sv, batch):
        self.message = message
        self.data_cv = data_cv
        self.data_sv = data_sv
        self.batch = batch
    def __str__(self):
        return self.message+' in the last batch with cv '+str(self.data_cv)+' samples, sv '+str(self.data_sv)+' samples and '+str(self.batch)+' batches.'

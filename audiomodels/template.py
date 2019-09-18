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


class NotDomainSlicerError(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message+' is not a object with DomainSlicer object as parent'

class WrongSlicerError(Exception):

    def __init__(self, slicer,message):
        self.slicer_name = slicer.__name__
        self.message = message

    def __str__(self):
        return self.slicer_name+' cannot be used as slicer '+self.message

class BatchLimitException(Exception):

    def __init__(self, message):
        self.message

    def __str__(self):
        return self.message

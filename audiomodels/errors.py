# -*- coding: utf-8 -*-

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

class UnknownArchBlockException(Exception):

    def __init__(self, arch_block_name):
        self.message=arch_block_name

    def __str__(self):
        return '{} not known architecture block in the Builder class, to use it please implement as recommended.'.format(self.message)

class UnknownInputError(Exception):

    def __init__(self, input):

        self.message=input

    def __str__(self):
        return '{} input not feedable for Neural Network block'.format(self.message)

class NoHyperparameterError(Exception):

    def __init__(self, tuning):

        self.message=tuning

    def __str__(self):
        return self.message

import sys
import copy
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from six import string_types
import argparse
import functools
from numpy import asarray, array, ravel, repeat, prod, mean, where, ones
#any all, function has it is in python 3, for any iterable, not only numpy arrays
from IPython.display import Math

def any(iterable):
    if hasattr(iterable,'__iter__'):
        for element in iterable:
            if element:
                return True
        return False
    else:
        return None

def all(iterable):
    if hasattr(iterable,'__iter__'):
        for element in iterable:
            if not element:
                return False
        return True

class AttrDict(dict):

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class ExtendList(list):

    @staticmethod
    def any(self, iterable=None):
        if iterable is not None:
            if hasattr(iterable,'__iter__'):
                for element in iterable:
                    if element:
                        return True
                return False
            else:
                return None
        elif hasattr(self,'__iter__'):
            for element in iterable:
                if element:
                    return True
            return False
        else:
            return None

    @staticmethod
    def all(self, iterable=None):
        if iterable is not None:
            if hasattr(iterable,'__iter__'):
                for element in iterable:
                    if not element:
                        return False
                return True
            else:
                return None
        elif hasattr(self,'__iter__'):
            for element in iterable:
                if not element:
                    return False
            return True

        return None

    def __lt__(self, other):
        return [True if a < other else False for a in self]

    def __gt__(self, other):
        return [True if a > other else False for a in self]

    def __ge__(self, other):
        return [True if a >= other else False for a in self]

    def __le__(self, other):
        return [True if a <= other else False for a in self]

    def __eq__(self, other):
        return [True if a == other else False for a in self]

    def __ne__(self, other):
        return [True if a != other else False for a in self]

    def __getitem__(self,index):
        if issubclass(index.__class__, int):
            return super( ExtendList, self).__getitem__(index)
        elif hasattr(index,'__iter__'):

            if len(index) == len(self):
                l = ExtendList([])
                for a in range(len(index)):
                    if issubclass(index[a].__class__,bool):
                        if index[a]:
                            l.append(self[a])
            return l

    def __setitem__(self, index, value):
        if issubclass(index.__class__, int):
            return super( ExtendList, self).__setitem__(index, value)
        elif hasattr(index,'__iter__'):

            if len(index) == len(self):
                for a in range(len(index)):
                    if issubclass(index[a].__class__,bool):
                        if index[a]:
                            super(ExtendList, self).__setitem__(a, value)

    def to_tuple(self):
        l = ExtendList([])
        for a in self:
            if issubclass(a.__class__,ExtendList):
                l.append(a.to_tuple())
            else:
                l.append(a)

        return tuple(l)






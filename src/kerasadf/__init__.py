"""Tensorflow/Keras implementation of Assumed Density Filtering (ADF) based
probabilistic neural networks.

This package provides implementations of several ADF based probabilistic
buildings blocks commonly used in neural networks. They are to be used within
the framework of Tesorflow/Keras. Unlike standard (deterministic) Keras layers
that propagate point estimates, the corresponding probabilsitic ADF layers
propagate a distribution parametrized by its mean and (co-)variance.

"""
from __future__ import absolute_import, division, print_function


# package meta data
__version__ = "19.2.0.dev0"  # 0Y.Minor.Micro CalVer format
__title__ = "keras-adf"
__description__ = "Assumed Density Filtering (ADF) Probabilistic Networks"
__url__ = "https://github.com/jmaces/keras-adf"
__uri__ = __url__
# __doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Jan Maces"
__email__ = "janmaces[at]gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright 2019 Jan Maces"

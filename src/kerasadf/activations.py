import six
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K


# private helper function for relu, pdf of standard normal distribution
def _gauss_density(x):
    return 1/np.sqrt(2*np.pi)*K.exp(-K.square(x)/2)


# private helper function for relu, cdf of standard normal distribution
def _gauss_cumulative(x):
    return 1/2*(1+math_ops.erf(x/np.sqrt(2)))


def relu(x, alpha=0.0, max_value=None, threshold=0.0, mode='diag'):
    """Rectified Linear Unit.

    Assumed Density Filtering (ADF) version of the Keras `relu` activation.

    Parameters
    ----------
    x : list or tuple
        Input tensors (means and covariances).
    alpha: float, optional
        Slope of negative section. Default is `0.0`.
        Currently no value other than the default is supported for  ADF.
    max_value: float, optional
        Saturation threshold. Default is `None`.
        Currently no value other than the default is supported for  ADF.
    threshold: float, optional
        Threshold value for thresholded activation. Default is `0.0`.
        Currently no value other than the default is supported for  ADF.
    mode: {"diag", "diagonal", "lowrank", "half", "full"}
        Covariance computation mode. Default is "diag".

    Returns
    -------
        List of transformed means and covariances, according to
        the ReLU activation: `max(x, 0)`.

    """
    if not alpha == 0.0:
        raise NotImplementedError(
            'The relu activation function with alpha other than 0.0 has'
            'not been implemented for ADF layers yet.'
        )
    if max_value is not None:
        raise NotImplementedError(
            'The relu activation function with max_value other than `None` has '
            'not been implemented for ADF layers yet.'
        )
    if not threshold == 0.0:
        raise NotImplementedError(
            'The relu activation function with threshold other than 0.0 has'
            'not been implemented for ADF layers yet.'
        )
    if not isinstance(x, list) and len(x) == 2:
        raise ValueError('The relu activation function expects a list of '
                         'exactly two input tensors, but got: %s' % x)
    means, covariances = x
    means_shape = means.get_shape().as_list()
    means_rank = len(means_shape)
    cov_shape = covariances.get_shape().as_list()
    cov_rank = len(cov_shape)
    # treat inputs according to rank and mode
    if means_rank == 1:
        # if rank(mean)=1, treat as single vector, no reshapes necessary
        pass
    elif means_rank == 2:
        # if rank(mean)=2, treat as batch of vectors, no reshapes necessary
        pass
    else:
        # if rank(mean)=2+n, treat as batch of rank=n tensors + channels
        means = K.reshape(
            means,
            [-1] + [K.prod(means_shape[1:])],
        )
        if mode == 'diag':
            covariances = K.reshape(
                covariances,
                [-1] + [K.prod(cov_shape[1:])],
            )
        elif mode == 'half':
            covariances = K.reshape(
                covariances,
                [-1] + [cov_shape[1]] + [K.prod(cov_shape[2:])],
            )
        elif mode == 'full':
            covariances = K.reshape(
                covariances,
                [-1]
                + [K.prod(cov_shape[1:(cov_rank-1)//2+1])]
                + [K.prod(cov_shape[(cov_rank-1)//2+1:])],
            )

    if mode == 'diag':
        covariances = covariances + K.epsilon()
        std = K.sqrt(covariances)
        div = means/std
        gd_div = _gauss_density(div)
        gc_div = _gauss_cumulative(div)
        new_means = means*gc_div + std*gd_div
        new_covariances = (
            (K.square(means)+covariances)*gc_div
            + means*std*gd_div
            - K.square(new_means)
        )
        new_covariances = K.maximum(0.0, new_covariances)
    elif mode == 'half':
        variances = math_ops.reduce_sum(K.square(covariances), axis=1) + K.epsilon()
        std = K.sqrt(variances)
        div = means/std
        gd_div = _gauss_density(div)
        gc_div = _gauss_cumulative(div)
        new_means = means*gc_div + std*gd_div
        gc_div = K.expand_dims(gc_div, 1)
        new_covariances = covariances * gc_div
    elif mode == 'full':
        variances = array_ops.matrix_diag_part(covariances) + 1e-7
        std = K.sqrt(variances)
        div = means/std
        gd_div = _gauss_density(div)
        gc_div = _gauss_cumulative(div)
        new_means = means*gc_div + std*gd_div
        gc_div = K.expand_dims(gc_div, 1)
        new_covariances = covariances * gc_div
        new_covariances = K.permute_dimensions(new_covariances, [0, 2, 1])
        new_covariances = new_covariances * gc_div
        new_covariances = K.permute_dimensions(new_covariances, [0, 2, 1])
    # undo reshapes if necessary
    new_means = K.reshape(new_means, [-1] + means_shape[1:])
    new_covariances = K.reshape(new_covariances, [-1] + cov_shape[1:])
    return [new_means, new_covariances]


def linear(x, mode='diag'):
    """Linear (identity) activation function.

    Assumed Density Filtering (ADF) version of the Keras `linear` activation.

    Parameters
    ----------
    x : list or tuple
        Input tensors (means and covariances).
    mode: {"diag", "diagonal", "lowrank", "half", "full"}
        Covariance computation mode (Has no effect for linear activation).

    Returns
    -------
        List of transformed means and covariances, according to
        the linear identity activation: `x`.

    """
    means, covariances = x
    return [means, covariances]


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    if custom_objects and name in custom_objects:
        fn = custom_objects.get(name)
    else:
        fn = globals().get(name)
        if fn is None:
            print('WARNING: Unknown activation function: ' + name
                  + ' It has been replaced by the linear identity activation')
            fn = linear
    return fn


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret activation function identifier:',
                         identifier)

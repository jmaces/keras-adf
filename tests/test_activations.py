"""Tests for `kerasadf.activations`. """
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from hypothesis import given
from tensorflow.keras import backend as K

import kerasadf.activations
import kerasadf.layers

from .strategies import assert_eq, assert_leq, batched_float_array


# constants for various tests
ALL_ACTIVATIONS = ["linear", "relu"]


# serialization test
@pytest.mark.parametrize("name", ALL_ACTIVATIONS)
def test_serialization(name):
    K.clear_session()
    fn = kerasadf.activations.get(name)
    ref_fn = getattr(kerasadf.activations, name)
    assert fn == ref_fn
    config = kerasadf.activations.serialize(fn)
    fn = kerasadf.activations.deserialize(config)
    assert fn == ref_fn


@pytest.mark.parametrize("name", ALL_ACTIVATIONS)
def test_serialization_with_layers(name):
    K.clear_session()
    activation = kerasadf.activations.get(name)
    layer_from_name = kerasadf.layers.Dense(3, activation=name)
    layer_from_activation = kerasadf.layers.Dense(3, activation=activation)
    config_from_name = kerasadf.layers.serialize(layer_from_name)
    config_from_activation = kerasadf.layers.serialize(layer_from_activation)
    deserialized_layer_from_name = kerasadf.layers.deserialize(
        config_from_name
    )
    deserialized_layer_from_activation = kerasadf.layers.deserialize(
        config_from_activation
    )
    assert (
        deserialized_layer_from_name.__class__.__name__
        == layer_from_name.__class__.__name__
    )
    assert (
        deserialized_layer_from_activation.__class__.__name__
        == layer_from_activation.__class__.__name__
    )
    assert (
        deserialized_layer_from_name.__class__.__name__
        == layer_from_activation.__class__.__name__
    )
    assert (
        deserialized_layer_from_activation.__class__.__name__
        == layer_from_name.__class__.__name__
    )
    assert deserialized_layer_from_name.activation == activation
    assert deserialized_layer_from_activation.activation == activation


# activation tests
@given(batched_float_array())
def test_linear_eq_np_linear(x):
    def _np_linear(means, covariances, mode):
        return means, covariances

    K.clear_session()
    means, covariances, mode = x
    means_out, covariances_out = kerasadf.activations.linear(
        [means, covariances], mode=mode
    )
    means_ref, covariances_ref = _np_linear(means, covariances, mode)
    assert_eq(means_ref, means_out)
    assert_eq(covariances_ref, covariances_out)


@given(batched_float_array())
def test_relu(x):
    K.clear_session()
    means, covariances, mode = x
    means_tensor = K.placeholder(means.shape, dtype=means.dtype)
    covariances_tensor = K.placeholder(
        covariances.shape, dtype=covariances.dtype
    )
    f = K.function(
        [means_tensor, covariances_tensor],
        kerasadf.activations.relu(
            [means_tensor, covariances_tensor], mode=mode
        ),
    )
    means_out, covariances_out = f([means, covariances])
    assert means.shape == means_out.shape
    assert covariances.shape == covariances_out.shape
    assert means.dtype.name == means_out.dtype.name
    assert covariances.dtype.name == covariances_out.dtype.name
    assert_leq(np.zeros_like(means_out), means_out)
    assert_leq(means, means_out)
    if mode == "diag":
        variances_out = covariances_out
    elif mode == "half":
        cov_shape = covariances_out.shape
        variances_out = np.reshape(
            np.sum(
                np.square(
                    np.reshape(
                        covariances_out,
                        (cov_shape[0], cov_shape[1], np.prod(cov_shape[2:])),
                    )
                ),
                axis=1,
            ),
            means_out.shape,
        )
    elif mode == "full":
        cov_shape = covariances_out.shape
        cov_rank = len(cov_shape) - 1
        variances_out = np.reshape(
            np.diagonal(
                np.reshape(
                    covariances_out,
                    (
                        cov_shape[0],
                        np.prod(cov_shape[1 : cov_rank // 2 + 1]),
                        np.prod(cov_shape[cov_rank // 2 + 1 :]),
                    ),
                ),
                axis1=-2,
                axis2=-1,
            ),
            means_out.shape,
        )
    assert means_out.shape == variances_out.shape
    assert_leq(np.zeros_like(variances_out), variances_out)

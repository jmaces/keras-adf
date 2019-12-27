"""Tests for `kerasadf.layers.core`. """
from __future__ import absolute_import, division, print_function

import hypothesis.strategies as st

from hypothesis import given, settings
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import kerasadf.layers

from .strategies import assert_eq, batched_float_array


# core layer tests
@settings(deadline=None)
@given(batched_float_array(min_data_dims=2))
def test_flatten(x):
    K.clear_session()
    means, covariances, mode = x
    assert mode in ["diag", "half", "full"]
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.Flatten(mode=mode)
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    assert means.shape[0] == means_out.shape[0]
    assert ms.as_list() == om.shape.as_list()
    assert covariances.shape[0] == covariances_out.shape[0]
    assert cs.as_list() == oc.shape.as_list()
    assert 2 == len(means_out.shape)
    if mode == "diag":
        assert 2 == len(covariances_out.shape)
    elif mode == "half":
        assert 3 == len(covariances_out.shape)
    elif mode == "full":
        assert 3 == len(covariances_out.shape)
    assert_eq(means.flatten(), means_out.flatten())
    assert_eq(covariances.flatten(), covariances_out.flatten())
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.Flatten.from_config(config)
    layer_deserialized = kerasadf.layers.deserialize(
        {"class_name": layer.__class__.__name__, "config": config}
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_from_config
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_deserialized
    )


@settings(deadline=None)
@given(
    batched_float_array(max_data_dims=1),
    st.integers(min_value=1, max_value=128),
)
def test_dense(x, units):
    K.clear_session()
    means, covariances, mode = x
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.Dense(units, mode=mode)
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    assert means.shape[:-1] == means_out.shape[:-1]
    assert units == means_out.shape[-1]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[:-1] == covariances_out.shape[:-1]
        assert units == covariances_out.shape[-1]
    elif mode == "half":
        assert covariances.shape[:-1] == covariances_out.shape[:-1]
        assert units == covariances_out.shape[-1]
    elif mode == "full":
        assert covariances.shape[:-2] == covariances_out.shape[:-2]
        assert units == covariances_out.shape[-1]
        assert units == covariances_out.shape[-2]
    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.Dense.from_config(config)
    layer_deserialized = kerasadf.layers.deserialize(
        {"class_name": layer.__class__.__name__, "config": config}
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_from_config
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_deserialized
    )

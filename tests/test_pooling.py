"""Tests for `kerasadf.layers.pooling`. """
import hypothesis.strategies as st
import numpy as np
import pytest

from hypothesis import given, settings
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

import kerasadf.layers

from .strategies import batched_float_array


# pooling layer tests
@settings(deadline=None)
@pytest.mark.parametrize("padding", ["same", "valid"])
@given(
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=2, max_data_dims=2),
)
def test_average_pool_1d(padding, pool_size, strides, x):
    K.clear_session()
    means, covariances, mode = x
    strides = min(strides, means.shape[1])
    pool_size = min(pool_size, means.shape[1])
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.AveragePooling1D(
        pool_size, strides, padding, mode=mode
    )
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    if padding == "same":
        out_size = np.ceil(means.shape[1] / strides)
    elif padding == "valid":
        out_size = np.ceil((means.shape[1] - pool_size + 1) / strides)
    assert means.shape[0] == means_out.shape[0]
    assert out_size == means_out.shape[1]
    assert means.shape[2] == means_out.shape[2]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size == covariances_out.shape[1]
        assert covariances.shape[2] == covariances_out.shape[2]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size == covariances_out.shape[2]
        assert covariances.shape[3] == covariances_out.shape[3]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size == covariances_out.shape[1]
        assert covariances.shape[2] == covariances_out.shape[2]
        assert out_size == covariances_out.shape[3]
        assert covariances.shape[4] == covariances_out.shape[4]

    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.AveragePooling1D.from_config(config)
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
@pytest.mark.parametrize("padding", ["same", "valid"])
@given(
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=3, max_data_dims=3),
)
def test_average_pool_2d(padding, pool_size, strides, x):
    K.clear_session()
    means, covariances, mode = x
    if isinstance(strides, tuple):
        strides = np.minimum(strides, means.shape[1:3])
    else:
        strides = min(strides, min(means.shape[1], means.shape[2]))
    if isinstance(pool_size, tuple):
        pool_size = np.minimum(pool_size, means.shape[1:3])
    else:
        pool_size = min(pool_size, min(means.shape[1], means.shape[2]))
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.AveragePooling2D(
        pool_size, strides, padding, mode=mode
    )
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    if padding == "same":
        out_size = np.ceil(np.asarray(means.shape[1:3]) / strides)
    elif padding == "valid":
        out_size = np.ceil(
            (np.asarray(means.shape[1:3]) - pool_size + 1) / strides
        )
    assert means.shape[0] == means_out.shape[0]
    assert out_size[0] == means_out.shape[1]
    assert out_size[1] == means_out.shape[2]
    assert means.shape[3] == means_out.shape[3]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size[0] == covariances_out.shape[1]
        assert out_size[1] == covariances_out.shape[2]
        assert covariances.shape[3] == covariances_out.shape[3]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size[0] == covariances_out.shape[2]
        assert out_size[1] == covariances_out.shape[3]
        assert covariances.shape[4] == covariances_out.shape[4]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size[0] == covariances_out.shape[1]
        assert out_size[1] == covariances_out.shape[2]
        assert covariances.shape[3] == covariances_out.shape[3]
        assert out_size[0] == covariances_out.shape[4]
        assert out_size[1] == covariances_out.shape[5]
        assert covariances.shape[6] == covariances_out.shape[6]

    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.AveragePooling2D.from_config(config)
    layer_deserialized = kerasadf.layers.deserialize(
        {"class_name": layer.__class__.__name__, "config": config}
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_from_config
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_deserialized
    )

"""Tests for `kerasadf.layers.convolutional`. """
from __future__ import absolute_import, division, print_function

import hypothesis.strategies as st
import numpy as np
import pytest

from hypothesis import given, settings
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import kerasadf.layers

from .strategies import batched_float_array


# convolution layer tests
@settings(deadline=None)
@pytest.mark.parametrize("padding", ["same", "valid"])
@given(
    st.integers(min_value=1, max_value=64),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=2, max_data_dims=2),
)
def test_convolution_1d(padding, filters, kernel_size, strides, x):
    K.clear_session()
    means, covariances, mode = x
    strides = min(strides, means.shape[1])
    kernel_size = min(kernel_size, means.shape[1])
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.Conv1D(
        filters, kernel_size, strides, padding, mode=mode
    )
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    if padding == "same":
        out_size = np.ceil(means.shape[1] / strides)
    elif padding == "valid":
        out_size = np.ceil((means.shape[1] - kernel_size + 1) / strides)
    assert means.shape[0] == means_out.shape[0]
    assert out_size == means_out.shape[1]
    assert filters == means_out.shape[2]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size == covariances_out.shape[1]
        assert filters == covariances_out.shape[2]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size == covariances_out.shape[2]
        assert filters == covariances_out.shape[3]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size == covariances_out.shape[1]
        assert filters == covariances_out.shape[2]
        assert out_size == covariances_out.shape[3]
        assert filters == covariances_out.shape[4]

    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.Conv1D.from_config(config)
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
    st.integers(min_value=1, max_value=64),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=2, max_data_dims=2),
)
def test_dilated_convolution_1d(
    padding, filters, kernel_size, dilation_rate, x
):
    K.clear_session()
    means, covariances, mode = x
    kernel_size = min(kernel_size, means.shape[1])
    if kernel_size > 1:
        dilation_rate = min(
            dilation_rate,
            int(np.floor((means.shape[1] - 1) / (kernel_size - 1))),
        )
    else:
        dilation_rate = min(dilation_rate, means.shape[1])
    print("kernel", kernel_size)
    print("dilation", dilation_rate)
    print("input", means.shape[1])
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.Conv1D(
        filters,
        kernel_size,
        1,
        padding,
        dilation_rate=dilation_rate,
        mode=mode,
    )
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    if padding == "same":
        out_size = means.shape[1]
    elif padding == "valid":
        out_size = np.ceil(
            (means.shape[1] - (kernel_size - 1) * dilation_rate)
        )
    assert means.shape[0] == means_out.shape[0]
    assert out_size == means_out.shape[1]
    assert filters == means_out.shape[2]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size == covariances_out.shape[1]
        assert filters == covariances_out.shape[2]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size == covariances_out.shape[2]
        assert filters == covariances_out.shape[3]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size == covariances_out.shape[1]
        assert filters == covariances_out.shape[2]
        assert out_size == covariances_out.shape[3]
        assert filters == covariances_out.shape[4]

    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.Conv1D.from_config(config)
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
    st.integers(min_value=1, max_value=64),
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
def test_convolution_2d(padding, filters, kernel_size, strides, x):
    K.clear_session()
    means, covariances, mode = x
    if isinstance(strides, tuple):
        strides = np.minimum(strides, means.shape[1:3])
    else:
        strides = min(strides, min(means.shape[1], means.shape[2]))
    if isinstance(kernel_size, tuple):
        kernel_size = np.minimum(kernel_size, means.shape[1:3])
    else:
        kernel_size = min(kernel_size, min(means.shape[1], means.shape[2]))
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.Conv2D(
        filters, kernel_size, strides, padding, mode=mode
    )
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    if padding == "same":
        out_size = np.ceil(np.asarray(means.shape[1:3]) / strides)
    elif padding == "valid":
        out_size = np.ceil(
            (np.asarray(means.shape[1:3]) - kernel_size + 1) / strides
        )
    assert means.shape[0] == means_out.shape[0]
    assert out_size[0] == means_out.shape[1]
    assert out_size[1] == means_out.shape[2]
    assert filters == means_out.shape[3]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size[0] == covariances_out.shape[1]
        assert out_size[1] == covariances_out.shape[2]
        assert filters == covariances_out.shape[3]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size[0] == covariances_out.shape[2]
        assert out_size[1] == covariances_out.shape[3]
        assert filters == covariances_out.shape[4]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size[0] == covariances_out.shape[1]
        assert out_size[1] == covariances_out.shape[2]
        assert filters == covariances_out.shape[3]
        assert out_size[0] == covariances_out.shape[4]
        assert out_size[1] == covariances_out.shape[5]
        assert filters == covariances_out.shape[6]

    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.Conv2D.from_config(config)
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
    st.integers(min_value=1, max_value=64),
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
def test_dilated_convolution_2d(
    padding, filters, kernel_size, dilation_rate, x
):
    K.clear_session()
    means, covariances, mode = x
    if isinstance(kernel_size, tuple):
        kernel_size = np.minimum(kernel_size, means.shape[1:3])
        kernel_size_tuple = np.asarray(kernel_size)
    else:
        kernel_size = min(kernel_size, min(means.shape[1], means.shape[2]))
        kernel_size_tuple = np.asarray((kernel_size, kernel_size))
    rate_limit = np.asarray(
        [
            int(np.floor(np.divide(means.shape[i + 1], kernel_size_tuple[i])))
            if means.shape[i + 1] > 1
            else np.inf
            for i in (0, 1)
        ]
    )
    if isinstance(dilation_rate, tuple):
        dilation_rate = np.minimum(
            dilation_rate, np.minimum(means.shape[1:3], rate_limit)
        )
    else:
        dilation_rate = int(
            min(dilation_rate, min(means.shape[1:3]), min(rate_limit))
        )
    im = Input(shape=means.shape[1:], dtype=means.dtype)
    ic = Input(shape=covariances.shape[1:], dtype=covariances.dtype)
    layer = kerasadf.layers.Conv2D(
        filters,
        kernel_size,
        1,
        padding,
        dilation_rate=dilation_rate,
        mode=mode,
    )
    ms, cs = layer.compute_output_shape([im.shape, ic.shape])
    om, oc = layer([im, ic])
    model = Model([im, ic], [om, oc])
    means_out, covariances_out = model.predict([means, covariances])
    if padding == "same":
        out_size = means.shape[1:3]
    elif padding == "valid":
        out_size = np.ceil(
            (np.asarray(means.shape[1:3]) - (kernel_size - 1) * dilation_rate)
        )
    assert means.shape[0] == means_out.shape[0]
    assert out_size[0] == means_out.shape[1]
    assert out_size[1] == means_out.shape[2]
    assert filters == means_out.shape[3]
    assert ms.as_list() == om.shape.as_list()
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size[0] == covariances_out.shape[1]
        assert out_size[1] == covariances_out.shape[2]
        assert filters == covariances_out.shape[3]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size[0] == covariances_out.shape[2]
        assert out_size[1] == covariances_out.shape[3]
        assert filters == covariances_out.shape[4]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert out_size[0] == covariances_out.shape[1]
        assert out_size[1] == covariances_out.shape[2]
        assert filters == covariances_out.shape[3]
        assert out_size[0] == covariances_out.shape[4]
        assert out_size[1] == covariances_out.shape[5]
        assert filters == covariances_out.shape[6]

    assert cs.as_list() == oc.shape.as_list()
    # serialization and deserialization test
    config = layer.get_config()
    layer_from_config = kerasadf.layers.Conv2D.from_config(config)
    layer_deserialized = kerasadf.layers.deserialize(
        {"class_name": layer.__class__.__name__, "config": config}
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_from_config
    )
    assert kerasadf.layers.serialize(layer) == kerasadf.layers.serialize(
        layer_deserialized
    )

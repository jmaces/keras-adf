import numpy as np
import tensorflow.python.keras.backend as K

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn
from tensorflow.python.util import nest

from .core import ADFLayer


class Pooling1D(ADFLayer):
    """Pooling layer for arbitrary pooling functions, for 1D inputs.

    Assumed Density Filtering (ADF) version of the abstract Keras ``Pooling1D``
    layer.

    This class only exists for code reuse. It will never be an exposed API.

    Parameters
    ----------
    pool_function : callable
        The pooling function to apply, e.g. ``tf.nn.max_pool2d``.
    pool_size : int or tuple of int
        An integer or tuple/list of a single integer, representing the size of
        the pooling window.
    strides : int or tuple of int or `None`, optional
        An integer or tuple/list of a single integer, specifying the strides of
        the pooling operation. If `None`, the ``pool_size`` will be used.
        Default is `None`.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, steps, features)`` while "channels_first" corresponds to
        inputs with shape ``(batch, features, steps)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".
    """

    def __init__(
        self,
        pool_function,
        pool_size,
        strides,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(Pooling1D, self).__init__(**kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 1, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.supports_masking = False
        if self.mode == "diag":
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        elif self.mode == "half":
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=4)]
        elif self.mode == "full":
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=5)]

    def call(self, inputs):
        raise NotImplementedError(
            "Pooling1D can not be called directly. The core functionality "
            "has to be implemented by a derived class."
        )

    def compute_output_shape(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == "channels_first":
            mean_steps = input_shape[0][2]
            if self.mode == "diag":
                cov_steps = input_shape[1][2]
            elif self.mode == "half":
                cov_steps = input_shape[1][3]
            elif self.mode == "full":
                cov_steps = input_shape[1][2]
        else:
            mean_steps = input_shape[0][1]
            if self.mode == "diag":
                cov_steps = input_shape[1][1]
            elif self.mode == "half":
                cov_steps = input_shape[1][2]
            elif self.mode == "full":
                cov_steps = input_shape[1][1]
        out_mean_steps = conv_utils.conv_output_length(
            mean_steps, self.pool_size[0], self.padding, self.strides[0]
        )
        out_cov_steps = conv_utils.conv_output_length(
            cov_steps, self.pool_size[0], self.padding, self.strides[0]
        )
        if self.data_format == "channels_first":
            if self.mode == "diag":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], input_shape[0][1], out_mean_steps]
                    ),
                    tensor_shape.TensorShape(
                        [input_shape[1][0], input_shape[1][1], out_cov_steps]
                    ),
                ]
            elif self.mode == "half":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], input_shape[0][1], out_mean_steps]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            input_shape[1][2],
                            out_cov_steps,
                        ]
                    ),
                ]
            elif self.mode == "full":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], input_shape[0][1], out_mean_steps]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            out_cov_steps,
                            input_shape[1][3],
                            out_cov_steps,
                        ]
                    ),
                ]
        else:
            if self.mode == "diag":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], out_mean_steps, input_shape[0][2]]
                    ),
                    tensor_shape.TensorShape(
                        [input_shape[1][0], out_cov_steps, input_shape[1][2]]
                    ),
                ]
            elif self.mode == "half":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], out_mean_steps, input_shape[0][2]]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            out_cov_steps,
                            input_shape[1][3],
                        ]
                    ),
                ]
            elif self.mode == "full":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], out_mean_steps, input_shape[0][2]]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            out_cov_steps,
                            input_shape[1][2],
                            out_cov_steps,
                            input_shape[1][4],
                        ]
                    ),
                ]

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(Pooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling1D(Pooling1D):
    """Max pooling layer for 1D inputs.

    Assumed Density Filtering (ADF) version of `keras.layers.MaxPooling1D`.

    Parameters
    ----------
    pool_size : int or tuple of int
        An integer or tuple/list of a single integer, representing the size of
        the pooling window. Default is 2.
    strides : int or tuple of int or `None`, optional
        An integer or tuple/list of a single integer, specifying the strides of
        the pooling operation. If `None`, the ``pool_size`` will be used.
        Default is `None`.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, steps, features)`` while "channels_first" corresponds to
        inputs with shape ``(batch, features, steps)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".


    Notes
    -----
    Input shape
        3D tensor with shape ``(batch_size, steps, features)`` if
        ``data_format`` is "channels_last" or shape
        ``(batch_size, features, steps)`` if ``data_format`` is
        "channels_first".

    Output shape
        3D tensor with shape ``(batch_size, pooled_steps, features)`` if
        ``data_format`` is "channels_last" or shape
        ``(batch_size, features, pooled_steps)`` if ``data_format`` is
        "channels_first".
    """

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        raise NotImplementedError(
            "MaxPooling1D has not been implemented as an ADF layer yet."
        )

    # super(MaxPooling1D, self).__init__(
    #     nn.max_pool,
    #     pool_size=pool_size, strides=strides,
    #     padding=padding, data_format=data_format, mode='half', **kwargs)

    def call(self, inputs):
        raise NotImplementedError(
            "MaxPooling1D has not been implemented as an ADF layer yet."
        )


class AveragePooling1D(Pooling1D):
    """Average pooling layer for 1D inputs.

    Assumed Density Filtering (ADF) version of `keras.layers.AveragePooling1D`.

    Parameters
    ----------
    pool_size : int or tuple of int
        An integer or tuple/list of a single integer, representing the size of
        the pooling window. Default is 2.
    strides : int or tuple of int or `None`, optional
        An integer or tuple/list of a single integer, specifying the strides of
        the pooling operation. If `None`, the ``pool_size`` will be used.
        Default is `None`.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, steps, features)`` while "channels_first" corresponds to
        inputs with shape ``(batch, features, steps)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".


    Notes
    -----
    Input shape
        3D tensor with shape ``(batch_size, steps, features)`` if
        ``data_format`` is "channels_last" or shape
        ``(batch_size, features, steps)`` if ``data_format`` is
        "channels_first".

    Output shape
        3D tensor with shape ``(batch_size, pooled_steps, features)`` if
        ``data_format`` is "channels_last" or shape
        ``(batch_size, features, pooled_steps)`` if ``data_format`` is
        "channels_first".
    """

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(AveragePooling1D, self).__init__(
            nn.avg_pool,  # no 1d pooling until tf-1.14, we use 2d instead
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )

    def call(self, inputs):
        input_shapes = nest.map_structure(lambda x: x.shape, inputs)
        output_shapes = self.compute_output_shape(input_shapes)
        means, covariances = inputs
        # there is no 1d pooling until tf-1.14, so we use 2d pooling instead
        if self.data_format == "channels_last":
            means = K.expand_dims(means, 1)
            if self.mode == "diag":
                covariances = K.expand_dims(covariances, 1)
            elif self.mode == "half":
                covariances = K.expand_dims(covariances, 2)
            elif self.mode == "full":
                covariances = K.expand_dims(covariances, 1)
                covariances = K.expand_dims(covariances, 4)
            pool_shape = list((1,) + self.pool_size)
            strides = list((1,) + self.strides)
            data_format = "NHWC"
        else:
            means = K.expand_dims(means, 2)
            if self.mode == "diag":
                covariances = K.expand_dims(covariances, 2)
            elif self.mode == "half":
                covariances = K.expand_dims(covariances, 3)
            elif self.mode == "full":
                covariances = K.expand_dims(covariances, 2)
                covariances = K.expand_dims(covariances, 5)
            pool_shape = list((1,) + self.pool_size)
            strides = list((1,) + self.strides)
            data_format = "NCHW"
        outputs = [[], []]
        outputs[0] = K.reshape(
            self.pool_function(
                means,
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper(),
                data_format=data_format,
            ),
            [-1] + output_shapes[0].as_list()[1:],
        )
        if self.mode == "diag":
            outputs[1] = K.reshape(
                self.pool_function(
                    covariances / np.prod(pool_shape),
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=data_format,
                ),
                [-1] + output_shapes[1].as_list()[1:],
            )
        elif self.mode == "half":
            cov_shape = covariances.get_shape().as_list()
            covariances = K.reshape(covariances, [-1] + cov_shape[2:])
            outputs[1] = K.reshape(
                self.pool_function(
                    covariances,
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=data_format,
                ),
                [-1] + output_shapes[1].as_list()[1:],
            )
        elif self.mode == "full":
            cov_shape = covariances.get_shape().as_list()
            out_shape = output_shapes[1].as_list()
            if self.data_format == "channels_last":
                out_shape = (
                    out_shape[:1] + [1] + out_shape[1:3] + [1] + out_shape[3:]
                )
            elif self.data_format == "channels_first":
                out_shape = (
                    out_shape[:2] + [1] + out_shape[2:4] + [1] + out_shape[4:]
                )
            covariances = K.reshape(covariances, [-1] + cov_shape[4:])
            covariances = K.reshape(
                self.pool_function(
                    covariances,
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=data_format,
                ),
                ([-1] + cov_shape[1:4] + out_shape[-3:]),
            )
            covariances = K.permute_dimensions(
                covariances, ([0] + list(range(4, 7)) + list(range(1, 4))),
            )
            covariances = K.reshape(covariances, [-1] + cov_shape[1:4])
            covariances = K.reshape(
                self.pool_function(
                    covariances,
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=data_format,
                ),
                ([-1] + out_shape[-3:] + out_shape[1:4]),
            )
            outputs[1] = K.reshape(
                K.permute_dimensions(
                    covariances, ([0] + list(range(4, 7)) + list(range(1, 4))),
                ),
                [-1] + output_shapes[1].as_list()[1:],
            )
        return outputs


class Pooling2D(ADFLayer):
    """Pooling layer for arbitrary pooling functions, for 2D inputs.

    Assumed Density Filtering (ADF) version of the abstract Keras ``Pooling2D``
    layer.

    This class only exists for code reuse. It will never be an exposed API.

    Parameters
    ----------
    pool_function : callable
        The pooling function to apply, e.g. ``tf.nn.max_pool2d``.
    pool_size : int or tuple of int
        An integer or tuple/list of two integers, ``(pool_height, pool_width)``
        specifying the size of the pooling window. Can be a single integer to
        specify the same value for all spatial dimensions.
    strides : int or tuple of int or `None`
        An integer or tuple/list of two integers, specifying the strides of
        the pooling operation. Can be a single integer to specify the same
        value for all spatial dimensions. If `None`, the ``pool_size`` will be
        used. Default is `None`.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, height, width, channels)`` while "channels_first" corresponds
        to inputs with shape ``(batch, channels, height, width)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".
    """

    def __init__(
        self,
        pool_function,
        pool_size,
        strides,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(Pooling2D, self).__init__(**kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.supports_masking = False
        if self.mode == "diag":
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        elif self.mode == "half":
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=5)]
        elif self.mode == "full":
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=7)]

    def call(self, inputs):
        raise NotImplementedError(
            "Pooling2D can not be called directly. The core functionality "
            "has to be implemented by a derived class."
        )

    def compute_output_shape(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == "channels_first":
            mean_rows = input_shape[0][2]
            mean_cols = input_shape[0][3]
            if self.mode == "diag":
                cov_rows = input_shape[1][2]
                cov_cols = input_shape[1][3]
            elif self.mode == "half":
                cov_rows = input_shape[1][3]
                cov_cols = input_shape[1][4]
            elif self.mode == "full":
                cov_rows = input_shape[1][2]
                cov_cols = input_shape[1][3]
        else:
            mean_rows = input_shape[0][1]
            mean_cols = input_shape[0][2]
            if self.mode == "diag":
                cov_rows = input_shape[1][1]
                cov_cols = input_shape[1][2]
            elif self.mode == "half":
                cov_rows = input_shape[1][2]
                cov_cols = input_shape[1][3]
            elif self.mode == "full":
                cov_rows = input_shape[1][1]
                cov_cols = input_shape[1][2]
        out_mean_rows = conv_utils.conv_output_length(
            mean_rows, self.pool_size[0], self.padding, self.strides[0]
        )
        out_mean_cols = conv_utils.conv_output_length(
            mean_cols, self.pool_size[1], self.padding, self.strides[1]
        )
        out_cov_rows = conv_utils.conv_output_length(
            cov_rows, self.pool_size[0], self.padding, self.strides[0]
        )
        out_cov_cols = conv_utils.conv_output_length(
            cov_cols, self.pool_size[1], self.padding, self.strides[1]
        )
        if self.data_format == "channels_first":
            if self.mode == "diag":
                return [
                    tensor_shape.TensorShape(
                        [
                            input_shape[0][0],
                            input_shape[0][1],
                            out_mean_rows,
                            out_mean_cols,
                        ]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            out_cov_rows,
                            out_cov_cols,
                        ]
                    ),
                ]
            elif self.mode == "half":
                return [
                    tensor_shape.TensorShape(
                        [
                            input_shape[0][0],
                            input_shape[0][1],
                            out_mean_rows,
                            out_mean_cols,
                        ]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            input_shape[1][2],
                            out_cov_rows,
                            out_cov_cols,
                        ]
                    ),
                ]
            elif self.mode == "full":
                return [
                    tensor_shape.TensorShape(
                        [
                            input_shape[0][0],
                            input_shape[0][1],
                            out_mean_rows,
                            out_mean_cols,
                        ]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            out_cov_rows,
                            out_cov_cols,
                            input_shape[1][4],
                            out_cov_rows,
                            out_cov_cols,
                        ]
                    ),
                ]
        else:
            if self.mode == "diag":
                return [
                    tensor_shape.TensorShape(
                        [
                            input_shape[0][0],
                            out_mean_rows,
                            out_mean_cols,
                            input_shape[0][3],
                        ]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            out_cov_rows,
                            out_cov_cols,
                            input_shape[1][3],
                        ]
                    ),
                ]
            elif self.mode == "half":
                return [
                    tensor_shape.TensorShape(
                        [
                            input_shape[0][0],
                            out_mean_rows,
                            out_mean_cols,
                            input_shape[0][3],
                        ]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            input_shape[1][1],
                            out_cov_rows,
                            out_cov_cols,
                            input_shape[1][4],
                        ]
                    ),
                ]
            elif self.mode == "full":
                return [
                    tensor_shape.TensorShape(
                        [
                            input_shape[0][0],
                            out_mean_rows,
                            out_mean_cols,
                            input_shape[0][3],
                        ]
                    ),
                    tensor_shape.TensorShape(
                        [
                            input_shape[1][0],
                            out_cov_rows,
                            out_cov_cols,
                            input_shape[1][3],
                            out_cov_rows,
                            out_cov_cols,
                            input_shape[1][6],
                        ]
                    ),
                ]

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Pooling2D):
    """Max pooling layer for 2D inputs.

    Assumed Density Filtering (ADF) version of `keras.layers.MaxPooling2D`.

    Parameters
    ----------
    pool_size : int or tuple of int, optional
        An integer or tuple/list of two integers, ``(pool_height, pool_width)``
        specifying the size of the pooling window. Can be a single integer to
        specify the same value for all spatial dimensions.
        Default is ``(2,2)``.
    strides : int or tuple of int or `None`
        An integer or tuple/list of two integers, specifying the strides of
        the pooling operation. Can be a single integer to specify the same
        value for all spatial dimensions. If `None`, the ``pool_size`` will be
        used. Default is `None`.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, height, width, channels)`` while "channels_first" corresponds
        to inputs with shape ``(batch, channels, height, width)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".


    Notes
    -----
    Input shape
        4D tensor with shape ``(batch_size, rows, cols, channels)`` if
        ``data_format`` is "channels_last" or shape
        ``(batch_size, channels, rows, cols)`` if ``data_format`` is
        "channels_first".

    Output shape
        4D tensor with shape
        ``(batch_size, pooled_rows, pooled_cols, channels)`` if ``data_format``
        is "channels_last" or shape
        ``(batch_size, channels, pooled_rows, pooled_cols)`` if ``data_format``
        is "channels_first".
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        raise NotImplementedError(
            "MaxPooling2D has not been implemented as an ADF layer yet."
        )

    # super(MaxPooling2D, self).__init__(
    #     nn.max_pool,
    #     pool_size=pool_size, strides=strides,
    #     padding=padding, data_format=data_format, mode='half', **kwargs)

    def call(self, inputs):
        raise NotImplementedError(
            "MaxPooling2D has not been implemented as an ADF layer yet."
        )


class AveragePooling2D(Pooling2D):
    """Average pooling layer for 2D inputs.

    Assumed Density Filtering (ADF) version of `keras.layers.AveragePooling2D`.

    Parameters
    ----------
    pool_size : int or tuple of int, optional
        An integer or tuple/list of two integers, ``(pool_height, pool_width)``
        specifying the size of the pooling window. Can be a single integer to
        specify the same value for all spatial dimensions.
        Default is ``(2,2)``.
    strides : int or tuple of int or `None`
        An integer or tuple/list of two integers, specifying the strides of
        the pooling operation. Can be a single integer to specify the same
        value for all spatial dimensions. If `None`, the ``pool_size`` will be
        used. Default is `None`.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, height, width, channels)`` while "channels_first" corresponds
        to inputs with shape ``(batch, channels, height, width)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".


    Notes
    -----
    Input shape
        4D tensor with shape ``(batch_size, rows, cols, channels)`` if
        ``data_format`` is "channels_last" or shape
        ``(batch_size, channels, rows, cols)`` if ``data_format`` is
        "channels_first".

    Output shape
        4D tensor with shape
        ``(batch_size, pooled_rows, pooled_cols, channels)`` if ``data_format``
        is "channels_last" or shape
        ``(batch_size, channels, pooled_rows, pooled_cols)`` if ``data_format``
        is "channels_first".
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(AveragePooling2D, self).__init__(
            nn.avg_pool,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )

    def call(self, inputs):
        input_shapes = nest.map_structure(lambda x: x.shape, inputs)
        output_shapes = self.compute_output_shape(input_shapes)
        means, covariances = inputs
        if self.data_format == "channels_last":
            pool_shape = list(self.pool_size)
            strides = list(self.strides)
        else:
            pool_shape = list(self.pool_size)
            strides = list(self.strides)
        outputs = [[], []]
        outputs[0] = self.pool_function(
            means,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        if self.mode == "diag":
            outputs[1] = self.pool_function(
                covariances / np.prod(pool_shape),
                ksize=pool_shape,
                strides=strides,
                padding=self.padding.upper(),
                data_format=conv_utils.convert_data_format(
                    self.data_format, 4
                ),
            )
        elif self.mode == "half":
            cov_shape = covariances.get_shape().as_list()
            covariances = K.reshape(covariances, [-1] + cov_shape[2:])
            outputs[1] = K.reshape(
                self.pool_function(
                    covariances,
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=conv_utils.convert_data_format(
                        self.data_format, 4
                    ),
                ),
                [-1] + output_shapes[1].as_list()[1:],
            )
        elif self.mode == "full":
            cov_shape = covariances.get_shape().as_list()
            covariances = K.reshape(covariances, [-1] + cov_shape[4:])
            covariances = K.reshape(
                self.pool_function(
                    covariances,
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=conv_utils.convert_data_format(
                        self.data_format, 4
                    ),
                ),
                ([-1] + cov_shape[1:4] + output_shapes[1].as_list()[-3:]),
            )
            covariances = K.permute_dimensions(
                covariances, ([0] + list(range(4, 7)) + list(range(1, 4))),
            )
            covariances = K.reshape(covariances, [-1] + cov_shape[1:4])
            covariances = K.reshape(
                self.pool_function(
                    covariances,
                    ksize=pool_shape,
                    strides=strides,
                    padding=self.padding.upper(),
                    data_format=conv_utils.convert_data_format(
                        self.data_format, 4
                    ),
                ),
                (
                    [-1]
                    + output_shapes[1].as_list()[-3:]
                    + output_shapes[1].as_list()[1:4]
                ),
            )
            outputs[1] = K.permute_dimensions(
                covariances, ([0] + list(range(4, 7)) + list(range(1, 4))),
            )
        return outputs

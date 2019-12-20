from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest

from .. import activations
from .core import ADFLayer


class Conv(ADFLayer):
    """Abstract nD convolution layer (private, used as implementation base).

    Assumed Density Filtering (ADF) version of the abstract Keras ``Conv``
    layer.


    Parameters
    ----------
    rank : int
        Rank of the convolution, e.g. "2" for 2D convolution.
    filters : int
        Dimensionality of the output space (i.e. the number of filters in
        the convolution).
    kernel_size : int or tuple of int or list of int
        An integer or tuple/list of n integers, specifying the length of the
        convolution window.
    strides : int or tuple of int or list of int, optional
        An integer or tuple/list of n integers, specifying the stride length of
        the convolution. Specifying any stride value != 1 is incompatible with
        specifying any ``dilation_rate`` value != 1. Default is 1.
    padding : {"valid", "same"}, optional
        The padding method. Case-insensitive. Default is "valid".
    data_format : {"channels_last", "channels_first"}, optional
        The ordering of the dimensions in the inputs.
        "channels_last" corresponds to inputs with shape
        ``(batch, ..., channels)`` while "channels_first" corresponds to
        inputs with shape ``(batch, channels, ...)``.
        It defaults to the ``image_data_format`` value found in your
        Keras config file at ``~/.keras/keras.json``.
        If you never set it, then it will be "channels_last".
    dilation_rate : int or tuple of int or list of int, optional
        An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any ``dilation_rate`` value != 1 is
        incompatible with specifying any ``strides`` value != 1. Default is 1.
    activation : callable or string, optional
        Activation function to use. Default is no activation
        (ie. "linear" activation: ``a(x) = x``).
    use_bias : bool
        Whether the layer uses a bias.
    kernel_initializer : Initializer or string, optional
        Initializer for the convolution ``kernel``.
        Default is "glorot_uniform" initialization.
    bias_initializer : Initializer or string, optional
        Initializer for the bias vector. Default is `None`.
    kernel_regularizer : Regularizer or string, optional
        Regularizer function applied to the convolution ``kernel``.
        Default is `None`.
    bias_regularizer : Regularizer or string, optional
        Regularizer function applied to the bias vector.
        Default is `None`.
    activity_regularizer : Regularizer or string, optional
        Regularizer function applied to the output of the layer.
        Default is `None`.
    kernel_constraint : Constraint or string, optional
        Constraint function applied to the convolution ``kernel``.
        Default is `None`.
    bias_constraint : Constraint or string, optional
        Constraint function applied to the bias vector.
        Default is `None`.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs
        )
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, "kernel_size"
        )
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False
        if self.mode == "diag":
            self.input_spec = [
                InputSpec(ndim=self.rank + 2),
                InputSpec(ndim=self.rank + 2),
            ]
        elif self.mode == "half":
            self.input_spec = [
                InputSpec(ndim=self.rank + 2),
                InputSpec(ndim=self.rank + 3),
            ]
        elif self.mode == "full":
            self.input_spec = [
                InputSpec(ndim=self.rank + 2),
                InputSpec(ndim=2 * self.rank + 3),
            ]

    def build(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0])
        input_shape[1] = tensor_shape.TensorShape(input_shape[1])
        if self.data_format == "channels_first":
            mean_channel_axis = 1
            if self.mode == "diag":
                cov_channel_axis = 1
            elif self.mode == "half":
                cov_channel_axis = 2
            elif self.mode == "full":
                cov_channel_axis = [1, 2 + self.rank]
        else:
            mean_channel_axis = -1
            if self.mode == "diag":
                cov_channel_axis = -1
            elif self.mode == "half":
                cov_channel_axis = -1
            elif self.mode == "full":
                cov_channel_axis = [-1, -self.rank - 2]
        if input_shape[0][mean_channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        if self.mode == "diag" and (
            input_shape[1][cov_channel_axis].value is None
        ):
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        if self.mode == "half" and (
            input_shape[1][cov_channel_axis].value is None
        ):
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        if self.mode == "full" and (
            input_shape[1][cov_channel_axis[0]].value is None
            or input_shape[1][cov_channel_axis[1]].value is None
        ):
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = int(input_shape[0][mean_channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_variable(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_variable(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        if self.mode == "diag":
            self.input_spec = [
                InputSpec(
                    ndim=self.rank + 2, axes={mean_channel_axis: input_dim}
                ),
                InputSpec(
                    ndim=self.rank + 2, axes={cov_channel_axis: input_dim}
                ),
            ]
        elif self.mode == "half":
            self.input_spec = [
                InputSpec(
                    ndim=self.rank + 2, axes={mean_channel_axis: input_dim}
                ),
                InputSpec(
                    ndim=self.rank + 3, axes={cov_channel_axis: input_dim}
                ),
            ]
        elif self.mode == "full":
            self.input_spec = [
                InputSpec(
                    ndim=self.rank + 2, axes={mean_channel_axis: input_dim}
                ),
                InputSpec(
                    ndim=2 * self.rank + 3,
                    axes={
                        cov_channel_axis[0]: input_dim,
                        cov_channel_axis[1]: input_dim,
                    },
                ),
            ]
        self._convolution_op = nn_ops.Convolution(
            input_shape[0],
            filter_shape=self.kernel.get_shape(),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(
                self.data_format, self.rank + 2
            ),
        )
        self.built = True

    def call(self, inputs):
        input_shapes = nest.map_structure(lambda x: x.shape, inputs)
        output_shapes = self.compute_output_shape(input_shapes)
        means, covariances = inputs
        outputs = [[], []]
        outputs[0] = self._convolution_op(means, self.kernel)
        if self.mode == "diag":
            outputs[1] = self._convolution_op(
                covariances, K.square(self.kernel)
            )
        elif self.mode == "half":
            cov_shape = covariances.get_shape().as_list()
            covariances = K.reshape(covariances, [-1] + cov_shape[2:])
            outputs[1] = K.reshape(
                self._convolution_op(covariances, self.kernel),
                [-1] + output_shapes[1].as_list()[1:],
            )
        elif self.mode == "full":
            cov_shape = covariances.get_shape().as_list()
            covariances = K.reshape(
                covariances, [-1] + cov_shape[self.rank + 2 :]
            )
            covariances = K.reshape(
                self._convolution_op(covariances, self.kernel),
                (
                    [-1]
                    + cov_shape[1 : self.rank + 2]
                    + output_shapes[1].as_list()[-self.rank - 1 :]
                ),
            )
            covariances = K.permute_dimensions(
                covariances,
                (
                    [0]
                    + list(range(self.rank + 2, 2 * self.rank + 3))
                    + list(range(1, self.rank + 2))
                ),
            )
            covariances = K.reshape(
                covariances, [-1] + cov_shape[1 : self.rank + 2]
            )
            covariances = K.reshape(
                self._convolution_op(covariances, self.kernel),
                (
                    [-1]
                    + output_shapes[1].as_list()[-self.rank - 1 :]
                    + output_shapes[1].as_list()[1 : self.rank + 2]
                ),
            )
            outputs[1] = K.permute_dimensions(
                covariances,
                (
                    [0]
                    + list(range(self.rank + 2, 2 * self.rank + 3))
                    + list(range(1, self.rank + 2))
                ),
            )

        if self.use_bias:
            outputs[0] = K.bias_add(
                outputs[0], self.bias, data_format=self.data_format
            )

        if self.activation is not None:
            return self.activation(outputs, mode=self.mode)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == "channels_last":
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            if self.mode == "diag":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0]] + new_space + [self.filters]
                    ),
                    tensor_shape.TensorShape(
                        [input_shape[1][0]] + new_space + [self.filters]
                    ),
                ]
            elif self.mode == "half":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0]] + new_space + [self.filters]
                    ),
                    tensor_shape.TensorShape(
                        input_shape[1][0:2] + new_space + [self.filters]
                    ),
                ]
            elif self.mode == "full":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0]] + new_space + [self.filters]
                    ),
                    tensor_shape.TensorShape(
                        [input_shape[1][0]]
                        + new_space
                        + [self.filters]
                        + new_space
                        + [self.filters]
                    ),
                ]
        else:
            space = input_shape[0][2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            if self.mode == "diag":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], self.filters] + new_space
                    ),
                    tensor_shape.TensorShape(
                        [input_shape[1][0], self.filters] + new_space
                    ),
                ]
            elif self.mode == "half":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], self.filters] + new_space
                    ),
                    tensor_shape.TensorShape(
                        input_shape[1][0:2] + [self.filters] + new_space
                    ),
                ]
            elif self.mode == "full":
                return [
                    tensor_shape.TensorShape(
                        [input_shape[0][0], self.filters] + new_space
                    ),
                    tensor_shape.TensorShape(
                        [input_shape[1][0], self.filters]
                        + new_space
                        + [self.filters]
                        + new_space
                    ),
                ]

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv1D(Conv):
    """1D convolution layer (for example temporal convolution).

    Assumed Density Filtering (ADF) version of `keras.layers.Conv1D`.


    Parameters
    ----------
    filters : int
        Dimensionality of the output space (i.e. the number of filters in
        the convolution).
    kernel_size : int or tuple of int or list of int
        An integer or tuple of int or list of a single integer, specifying the
        length of the convolution window.
    strides : int or tuple of int or list of int, optional
        An integer or tuple/list of a single integer, specifying the stride
        length of the convolution. Specifying any stride value != 1 is
        incompatible with specifying any ``dilation_rate`` value != 1.
        Default is 1.
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
    dilation_rate : int or tuple of int or list of int, optional
        An integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any ``dilation_rate`` value != 1 is
        incompatible with specifying any ``strides`` value != 1. Default is 1.
    activation : callable or string, optional
        Activation function to use. Default is no activation
        (ie. "linear" activation: ``a(x) = x``).
    use_bias : bool
        Whether the layer uses a bias.
    kernel_initializer : Initializer or string, optional
        Initializer for the convolution ``kernel``.
        Default is "glorot_uniform" initialization.
    bias_initializer : Initializer or string, optional
        Initializer for the bias vector. Default is `None`.
    kernel_regularizer : Regularizer or string, optional
        Regularizer function applied to the convolution ``kernel``.
        Default is `None`.
    bias_regularizer : Regularizer or string, optional
        Regularizer function applied to the bias vector.
        Default is `None`.
    activity_regularizer : Regularizer or string, optional
        Regularizer function applied to the output of the layer.
        Default is `None`.
    kernel_constraint : Constraint or string, optional
        Constraint function applied to the convolution ``kernel``.
        Default is `None`.
    bias_constraint : Constraint or string, optional
        Constraint function applied to the bias vector.
        Default is `None`.


    Notes
    -----
    Input shape
        3D tensor with shape ``(samples, features, steps)`` if ``data_format``
        is "channels_first" or shape ``(samples, steps, features)`` if
        ``data_format`` is "channels_last".

    Output shape
        3D tensor with shape ``(samples, filters, new_steps)`` if
        ``data_format`` is "channels_first" or shape
        ``(samples, new_steps, filters)`` if ``data_format`` is
        "channels_last".
        ``new_steps`` value might be different from ``steps`` due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )


class Conv2D(Conv):
    """2D convolution layer (for example spatial convolution).

    Assumed Density Filtering (ADF) version of `keras.layers.Conv2D`.


    Parameters
    ----------
    filters : int
        Dimensionality of the output space (i.e. the number of filters in
        the convolution).
    kernel_size : int or tuple of int or list of int
        An integer or tuple/list of two integers, specifying the width and
        height of the convolution window.
    strides : int or tuple of int or list of int, optional
        An integer or tuple/list of two integers, specifying the stride
        lengths of the convolution. Specifying any stride value != 1 is
        incompatible with specifying any ``dilation_rate`` value != 1.
        Default is 1.
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
    dilation_rate : int or tuple of int or list of int, optional
        An integer or tuple/list of two integers, specifying
        the dilation rates to use for dilated convolution.
        Currently, specifying any ``dilation_rate`` value != 1 is
        incompatible with specifying any ``strides`` value != 1. Default is 1.
    activation : callable or string, optional
        Activation function to use. Default is no activation
        (ie. "linear" activation: ``a(x) = x``).
    use_bias : bool
        Whether the layer uses a bias.
    kernel_initializer : Initializer or string, optional
        Initializer for the convolution ``kernel``.
        Default is "glorot_uniform" initialization.
    bias_initializer : Initializer or string, optional
        Initializer for the bias vector. Default is `None`.
    kernel_regularizer : Regularizer or string, optional
        Regularizer function applied to the convolution ``kernel``.
        Default is `None`.
    bias_regularizer : Regularizer or string, optional
        Regularizer function applied to the bias vector.
        Default is `None`.
    activity_regularizer : Regularizer or string, optional
        Regularizer function applied to the output of the layer.
        Default is `None`.
    kernel_constraint: Constraint or string, optional
        Constraint function applied to the convolution ``kernel``.
        Default is `None`.
    bias_constraint: Constraint or string, optional
        Constraint function applied to the bias vector.
        Default is `None`.


    Notes
    -----
    Input shape
        4D tensor with shape ``(samples, channels, height, width)`` if
        ``data_format`` is "channels_first" or shape
        ``(samples, height, width, channels)`` if ``data_format`` is
        "channels_last".

    Output shape
        4D tensor with shape ``(samples, filters, new_height, new_width)``
        if ``data_format`` is "channels_first" or shape
        ``(samples, new_height, new_width, filters)`` if ``data_format`` is
        "channels_last".
        ``new_height`` and ``new_width`` values might be different from
        ``height`` and ``width`` due to padding.

    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )

from tensorflow.python.keras.engine.base_layer import Layer, InputSpec
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
import numpy as np

from .. import activations


class ADFLayer(Layer):
    """Abstract base class for Assumed Density Filtering layers.

    Derived from the Keras Layer base class and offers the same functionality.
    It adds one additional required argument for all ADF layers, namely the
    mode of the probability distribution propagation through the layer.

    All modes propagate the *mean* (similar to the usual point-estimate of a
    standard Keras layer). Modes differ in how they treat second moment
    information, that is how they propagate correlations/covariances.

    ADF layers take two inputs and produce two outputs for every input and
    output of the respective standard Keras layer.

    Available modes are:

    1. `diag` or `diagonal`, meaning that nodes within a layer are considered
    statistically independent and only variances (but no covariances) are
    propagated. This amounts to propagating only the diagonal of the covariance
    matrix through the layer.

    2. `lowrank` or `half`, meaning that nodes within a layer are considered
    statistically dependent and variances as well as covariances are
    propagated. However, instead of the full covariance matrix only one half
    of a symmetric low-rank factorization of it is propagated for computational
    benefits.

    3. `full`, meaning that nodes within a layer are considered statistically
    dependent and variances as well as covariances are propagated. The full
    covariance matrix is propagated in this mode, which results in a huge
    memory requirement. Therefore, the full covariance mode is often infeasible
    and it is typically recommended to use the diagonal or low rank
    factorization mode instead.

    Attributes
    ----------
    mode : {'diag', 'half', 'full'}
        Covariance propagation mode.

    Parameters
    ----------
    mode : {'diag', 'diagonal', 'lowrank', 'half', 'full'}, optional
        Covariance propagation mode. The default mode is `diag`.
    **kwargs
        Keyword arguments, passed on to Keras `Layer` base class.

    """
    def __init__(self, mode='diag', **kwargs):
        # check and standardize mode parameter
        if mode in ['diag', 'diagonal']:
            mode = 'diag'
        elif mode in ['lowrank', 'half']:
            mode = 'half'
        elif mode in ['full']:
            mode = 'full'
        else:
            raise ValueError('Unknown covariance mode: {}'.format(mode))
        super(ADFLayer, self).__init__(**kwargs)
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(ADFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(ADFLayer):
    """Flattens the input. Does not affect the batch size.

    Assumed Density Filtering (ADF) version of the Keras `Flatten` layer.

    Parameters
    ----------
    data_format {'channels_last', 'channels_first'}, optional
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

    """
    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        if self.mode == 'diag':
            self.input_spec = [InputSpec(min_ndim=3), InputSpec(min_ndim=3)]
        elif self.mode == 'half':
            self.input_spec = [InputSpec(min_ndim=3), InputSpec(min_ndim=4)]
        elif self.mode == 'full':
            self.input_spec = [InputSpec(min_ndim=3), InputSpec(min_ndim=5)]

    def compute_output_shape(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        output_shape = [[input_shape[0][0]], [input_shape[1][0]]]
        if all(input_shape[0][1:]):
            output_shape[0] += [np.prod(input_shape[0][1:])]
        else:
            output_shape[0] += [None]
        if self.mode == 'diag':
            if all(input_shape[1][1:]):
                output_shape[1] += [np.prod(input_shape[1][1:])]
            else:
                output_shape[1] += [None]
        elif self.mode == 'half':
            if input_shape[1][1]:
                output_shape[1] += [input_shape[1][1]]
            else:
                output_shape[1] += [None]
            if all(input_shape[1][2:]):
                output_shape[1] += [np.prod(input_shape[1][2:])]
            else:
                output_shape[1] += [None]
        elif self.mode == 'full':
            if all(input_shape[1][1:(len(input_shape[1])-1)//2+1]):
                output_shape[1] += [np.prod(
                    input_shape[1][1:(len(input_shape[1])-1)//2+1])
                ]
            else:
                output_shape[1] += [None]
            if all(input_shape[1][(len(input_shape[1])-1)//2+1:]):
                output_shape[1] += [np.prod(
                    input_shape[1][(len(input_shape[1])-1)//2+1:])
                ]
            else:
                output_shape[1] += [None]

        return [
            tensor_shape.TensorShape(output_shape[0]),
            tensor_shape.TensorShape(output_shape[1]),
        ]

    def call(self, inputs):
        if self.data_format == 'channels_first':
            permutation = [[0], [0]]
            permutation[0].extend([i for i in
                                  range(2, K.ndim(inputs[0]))])
            permutation[0].append(1)
            inputs[0] = K.permute_dimensions(inputs[0], permutation[0])
            if self.mode == 'diag':
                permutation[1].extend([i for i in
                                      range(2, K.ndim(inputs[1]))])
                permutation[1].append(1)
                inputs[1] = K.permute_dimensions(inputs[1], permutation[1])
            elif self.mode == 'half':
                permutation[1].append(1)
                permutation[1].extend([i for i in
                                      range(3, K.ndim(inputs[1]))])
                permutation[1].append(2)
                inputs[1] = K.permute_dimensions(inputs[1], permutation[1])
            elif self.mode == 'full':
                permutation[1].extend([i for i in
                                      range(2, (K.ndim(inputs[1])-1)//2+1)])
                permutation[1].append(1)
                permutation[1].extend([i for i in
                                      range((K.ndim(inputs[1])-1)//2+2,
                                            K.ndim(inputs[1]))])
                permutation[1].append((K.ndim(inputs[1])-1)//2+1)
                inputs[1] = K.permute_dimensions(inputs[1], permutation[1])

        outputs = [[], []]
        outputs[0] = K.reshape(inputs[0],
                                       (K.shape(inputs[0])[0], -1))
        if self.mode == 'diag':
            outputs[1] = K.reshape(inputs[1],
                                           (K.shape(inputs[1])[0], -1))
        elif self.mode == 'half':
            outputs[1] = K.reshape(inputs[1],
                                           (K.shape(inputs[1])[0],
                                            K.shape(inputs[1])[1], -1))
        elif self.mode == 'full':
            outputs[1] = K.reshape(
                inputs[1],
                (
                    K.shape(inputs[1])[0],
                    K.prod(
                        K.shape(inputs[1])[
                            1:(K.ndim(inputs[1])-1)//2+1
                        ]
                    ),
                    K.prod(
                        K.shape(inputs[1])[
                            (K.ndim(inputs[1])-1)//2+1:K.ndim(inputs[1])
                        ]
                    ),
                )
            )

        if not context.executing_eagerly():
            out_shape = self.compute_output_shape(
                [inputs[0].get_shape(), inputs[1].get_shape()]
            )
            outputs[0].set_shape(out_shape[0])
            outputs[1].set_shape(out_shape[1])
        return outputs

    def get_config(self):
        config = {'data_format': self.data_format, 'mode': self.mode}
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense(Layer):
    """Densly-connected (fully connected) neural network layer.

    Assumed Density Filtering (ADF) version of the Keras `Dense` layer.

    Parameters
    ----------
    units : int
        Dimensionality of the output space (number of neurons).
    activation : callable or string, optional
        Activation function to use. Default is no activation
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    kernel_initializer : Initializer or string, optional
        Initializer for the `kernel` weights matrix.
        Default is `glorot_uniform` initialization.
    bias_initializer : Initializer or string, optional
        Initializer for the bias vector. Default is `None`.
    kernel_regularizer : Regularizer or string, optional
        Regularizer function applied to the `kernel` weights matrix.
        Default is `None`.
    bias_regularizer : Regulairzer or string, optional
        Regularizer function applied to the bias vector.
        Default is `None`.
    activity_regularizer : Regularizer or string, optional
        Regularizer function applied to the output of the layer.
        Default is `None`.
    kernel_constraint: Constraint or string, optional
        Constraint function applied to the `kernel` weights matrix.
        Default is `None`.
    bias_constraint: Constraint or string, optional
        Constraint function applied to the bias vector.
        Default is `None`.

    Notes
    -----
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        nD tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`.

    """
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

        if self.mode == 'diag':
            self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        elif self.mode == 'half':
            self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=3)]
        elif self.mode == 'full':
            self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=3)]

    def build(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0])
        input_shape[1] = tensor_shape.TensorShape(input_shape[1])
        if (input_shape[0][-1].value is None
                or input_shape[1][-1].value is None):
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = input_shape[0][-1].value
        if self.mode == 'diag':
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: last_dim}),
                InputSpec(min_ndim=2, axes={-1: last_dim}),
            ]
        elif self.mode == 'half':
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: last_dim}),
                InputSpec(min_ndim=3, axes={-1: last_dim}),
            ]
        elif self.mode == 'full':
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: last_dim}),
                InputSpec(min_ndim=3, axes={-1: last_dim, -2: last_dim}),
            ]
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        means, covariances = inputs
        outmeans = K.dot(means, self.kernel)
        if self.mode == 'diag':
            outcovariances = K.dot(covariances, K.square(self.kernel))
        elif self.mode == 'half':
            outcovariances = K.dot(covariances, self.kernel)
        elif self.mode == 'full':
            outcovariances = K.dot(covariances, self.kernel)
            outcovariances = K.dot(K.transpose(self.kernel), outcovariances)
        if self.use_bias:
            outmeans = K.bias_add(outmeans, self.bias)
        if self.activation is not None:
            return self.activation([outmeans, outcovariances], mode=self.mode)
        return [outmeans, outcovariances]

    def compute_output_shape(self, input_shape):
        input_shape[0] = tensor_shape.TensorShape(input_shape[0])
        input_shape[1] = tensor_shape.TensorShape(input_shape[1])
        input_shape[0] = input_shape[0].with_rank_at_least(2)
        if self.mode == 'diag':
            input_shape[1] = input_shape[1].with_rank_at_least(2)
        elif self.mode == 'half':
            input_shape[1] = input_shape[1].with_rank_at_least(3)
        elif self.mode == 'full':
            input_shape[1] = input_shape[1].with_rank_at_least(3)
        if input_shape[0][-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, '
                'but saw: %s' % input_shape[0])
        if input_shape[1][-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, '
                'but saw: %s' % input_shape[1])
        if ((self.mode == 'full' or self.mode == 'half')
                and input_shape[1][-2].value is None):
            raise ValueError(
                'The two innermost dimension of input_shape must be defined '
                'for modes "full" or "half", but saw: %s' % input_shape[1])
        if self.mode == 'diag' or self.mode == 'half':
            return [
                input_shape[0][:-1].concatenate(self.units),
                input_shape[1][:-1].concatenate(self.units),
            ]
        elif self.mode == 'full':
            return [
                input_shape[0][:-1].concatenate(self.units),
                input_shape[1][:-2].concatenate([self.units, self.units]),
            ]

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

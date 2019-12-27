Examples
========

Here we provide some more detailed examples showcasing some of the more advanced
aspects of using ``keras-adf``. For a simple use case and example of how to
get started we recommend to first have a look at our `overview` section.

.. contents:: Advanced topics addressed in our examples
    :depth: 1
    :local:
    :backlinks: none


Initializers, Regularizers, and Constraints
-------------------------------------------

Several ADF layers can handle different ``Initializers``, ``Regularizers``, and
``Constraints``, like their corresponding Keras layers. As an example we consider
the `Dense` layer, but the same concepts hold for the other layers.

Initializers for weights and biases can be passed either by name

.. doctest:: INIT_REGU_CONS

    >>> from kerasadf.layers import Dense
    >>> dense = Dense(16, kernel_initializer="glorot_uniform", bias_initializer="zero")

or as instances of the Keras ``Initializer`` class.

.. doctest:: INIT_REGU_CONS

    >>> from tensorflow.keras.initializers import Zeros, TruncatedNormal
    >>> dense = Dense(16, kernel_initializer=TruncatedNormal(0, 0.1), bias_initializer=Zeros())

In the same way ``Regularizers`` and ``Constraints`` are added to a layer.

.. doctest:: INIT_REGU_CONS

    >>> from tensorflow.keras.regularizers import L1L2
    >>> from tensorflow.keras.constraints import NonNeg
    >>> dense = Dense(16, kernel_regularizer=L1L2(l1=0.0, l2=0.1), bias_constraint=NonNeg())


(Co-)Variance Computation Modes
-------------------------------

There are three (co-)variance computation modes available for ADF layers.
All layers within a model must use the same mode to guarantee matching input/output shapes.

The three modes are:

    "diag" or "diagonal" mode
        This is the default for all layers. Here only
        variances but no covariances are propagated, i.e. in other words the dimensions
        of the inputs/activations/outputs are treated as independent/uncorrelated and only the
        diagonal of the covariance matrix is propagated. This independence is of course usually
        not really satisfied but in many scenarios a good enough approximation.
        In this mode the tensors for ``mean`` and ``variance`` in each layer have the same shape.
    "half" or "lowrank" mode
        This mode makes use of the symmetric factorization of the
        covariance matrix. Only one of the factors is propagated through the layers. The full output
        covariance matrix can be retrieved as the product of this factor with its transpose. Reducing the
        inner dimension of the matrix factors (which is kept constant throughout layers) allows the propagation
        of low-rank approximations to the covariance matrix and reduces the computational costs.
        For a mean tensor with shape ``(batch_size, num_dims)`` the corresponding covariance factor
        tensor must have shape ``(batch_size, rank, num_dims)``. In case of image data use
        ``(batch_size, rows, columns, channels)`` for the mean and ``(batch_size, rank, rows, columns, channels)``
        for the covariance factor.
    "full" mode
        This propagates the full covariance matrix and is computationally costly, in particular memory
        consumption can be problematic. Use this only for small layers and models. The covariance matrix requires the
        squared size of the mean tensor. For a mean tensor with shape ``(batch_size, num_dims)`` the corresponding covariance matrix
        tensor must have shape ``(batch_size, num_dims, num_dims)``. In case of image data use
        ``(batch_size, rows, columns, channels)`` for the mean and ``(batch_size, rows, columns, channels, rows, columns, channels)``
        for the covariance matrix.

As an example we will create `Dense` layers for all three modes. It works
analogously for all other layers.

.. doctest:: COV_MODES

    >>> from tensorflow.keras.layers import Input
    >>> from kerasadf.layers import Dense

First, we use the "diagonal" mode. Both inputs will need the same shape.

.. doctest:: COV_MODES

    >>> mean_in = Input((16,))
    >>> variance_in = Input((16,))
    >>> mean_out, variance_out = Dense(8, mode="diag")([mean_in, variance_in])
    >>> mean_out.shape
    TensorShape([Dimension(None), Dimension(8)])
    >>> variance_out.shape
    TensorShape([Dimension(None), Dimension(8)])


Next, we use the "half" mode. Here the second inputs need one additional
dimension for the rank of the matrix factorization.

.. doctest:: COV_MODES

    >>> rank = 4
    >>> mean_in = Input((16,))
    >>> covariance_factor_in = Input((rank, 16,))
    >>> mean_out, covariance_factor_out = Dense(8, mode="half")([mean_in, covariance_factor_in])
    >>> mean_out.shape
    TensorShape([Dimension(None), Dimension(8)])
    >>> covariance_factor_out.shape
    TensorShape([Dimension(None), Dimension(4), Dimension(8)])

Note that the rank dimension never changes and is passed on from inputs to outputs.
To obtain (the approximation to) the full output covariance matrix we compute the
tenor dot product of the factor with itself along the rank dimension.

.. doctest:: COV_MODES

    >>> from tensorflow.keras.backend import batch_dot
    >>> covariance_out = batch_dot(covariance_factor_out, covariance_factor_out, axes=(1, 1))
    >>> covariance_out.shape
    TensorShape([Dimension(None), Dimension(8), Dimension(8)])


Finally, we use the "full" mode. Here the second input requires the squared
size of the first input.

    .. doctest:: COV_MODES

        >>> mean_in = Input((16,))
        >>> covariance_in = Input((16, 16,))
        >>> mean_out, covariance_out = Dense(8, mode="full")([mean_in, covariance_in])
        >>> mean_out.shape
        TensorShape([Dimension(None), Dimension(8)])
        >>> covariance_out.shape
        TensorShape([Dimension(None), Dimension(8), Dimension(8)])

Overview
========

``keras-adf`` provides implementations of probabilistic assumed density filtering
(ADF) based version of layers commonly used in neural networks. They are to be
used within the `Tensorflow <https://www.tensorflow.org/>`_/`Keras <https://keras.io/>`_
framework. Unlike the standard (deterministic) Keras layers that propagate
point estimates, ADF layers propagate a probability distribution parametrized
by its mean and (co-)variance.

We think it is best to show the core concepts of the package by
a simple exemplary demonstration.

For this let us define a simple feed-forward model with fully-connected
ADF layers.

We begin by importing the relevant Keras and ``keras-adf`` layers.

.. testsetup:: OVERVIEW

    from tensorflow.keras.backend import clear_session
    clear_session()

.. doctest:: OVERVIEW

    >>> from tensorflow.keras.layers import Input
    >>> from tensorflow.keras.models import Model
    >>> from kerasadf.layers import Dense

Every model begins with its input. Since we propagate the mean and (co-)variance
through the ADF layers we need two inputs instead of just one.

.. doctest:: OVERVIEW

    >>> mean_in = Input((32,))
    >>> variance_in = Input((32,))

Next, we define a fully-connected hidden layer with rectified linear unit (ReLU)
activations and a fully-connected output layer with no activation.

.. doctest:: OVERVIEW

    >>> mean_hidden, variance_hidden = Dense(64, activation="relu")([mean_in, variance_in])
    >>> mean_out, variance_out = Dense(1)([mean_hidden, variance_hidden])

The complete model can now be assembled exactly like any other Keras model.

.. doctest:: OVERVIEW

    >>> model = Model([mean_in, variance_in], [mean_out, variance_out])

We have defined a dense feed-forward neural network with 32-dimensional input space,
a 64-dimensional hidden representation space, and a 1-dimensional output space.
The model summary looks like this:

.. doctest:: OVERVIEW

    >>> model.summary()  # doctest: +NORMALIZE_WHITESPACE
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            [(None, 32)]         0
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, 32)]         0
    __________________________________________________________________________________________________
    dense (Dense)                   [(None, 64), (None,  2112        input_1[0][0]
                                                                     input_2[0][0]
    __________________________________________________________________________________________________
    dense_1 (Dense)                 [(None, 1), (None, 1 65          dense[0][0]
                                                                     dense[0][1]
    ==================================================================================================
    Total params: 2,177
    Trainable params: 2,177
    Non-trainable params: 0
    __________________________________________________________________________________________________

This model can now be used like any other Keras model: It can be trained after
providing a loss function and an optimizer, it can be saved and restored, it
can be used to make predictions, ...

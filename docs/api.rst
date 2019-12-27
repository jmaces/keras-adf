API Reference
=============

.. automodule:: kerasadf

What follows is the *API explanation*. This mostly just lists functions and
their options and is intended for *quickly looking up* things.

If you like a more *hands-on introduction*, have a look at our `examples`.


kerasadf.activations
--------------------

.. automodule:: kerasadf.activations

Below is a list of supported activation functions in the package.

.. autosummary::
   :toctree: api

   linear
   relu


kerasadf.layers
---------------

.. automodule:: kerasadf.layers

Collects aliases of all supported layers in the package. See below for details.


kerasadf.layers.convolutional
-----------------------------

.. automodule:: kerasadf.layers.convolutional

Below is a list of supported convolutional layers in the package.
They have aliases making them directly available from :py:mod:`kerasadf.layers`.

.. autosummary::
   :toctree: api
   :template: class.rst


   Conv
   Conv1D
   Conv2D


kerasadf.layers.core
--------------------

.. automodule:: kerasadf.layers.core

Below is a list of supported core layers in the package.
They have aliases making them directly available from :py:mod:`kerasadf.layers`.

.. autosummary::
   :toctree: api
   :template: class.rst

   ADFLayer
   Dense
   Flatten


kerasadf.layers.pooling
-----------------------

.. automodule:: kerasadf.layers.pooling

Below is a list of supported pooling layers in the package.
They have aliases making them directly available from :py:mod:`kerasadf.layers`.

.. autosummary::
   :toctree: api
   :template: class.rst

   Pooling1D
   Pooling2D
   AveragePooling1D
   AveragePooling2D

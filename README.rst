============================================================================
``keras-adf``: Assumed Density Filtering (ADF) Probabilistic Neural Networks
============================================================================

.. add project badges here
.. image:: https://readthedocs.org/projects/keras-adf/badge/?version=latest
    :target: https://keras-adf.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/jmaces/keras-adf/actions/workflows/pr-check.yml/badge.svg?branch=master
    :target: https://github.com/jmaces/keras-adf/actions/workflows/pr-check.yml?branch=master
    :alt: CI Status

.. image:: https://codecov.io/gh/jmaces/keras-adf/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jmaces/keras-adf
  :alt: Code Coverage

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style: Black


.. teaser-start

``keras-adf`` provides implementations for probabilistic
`Tensorflow <https://www.tensorflow.org/>`_/`Keras <https://keras.io/>`_ neural network layers,
which are based on assumed density filtering.
Assumed density filtering (ADF) is a general concept from Bayesian inference, but in the case of feed-forward neural networks that we consider here
it is a way to approximately propagate a random distribution through the neural network.

The layers in this package have the same names and arguments as their corresponding
Keras version. We use Gaussian distributions for our ADF approximations, which are
described by their means and (co-)variances. So unlike the standard Keras layers,
each ``keras-adf`` layer takes two inputs and produces two outputs (one for the means
and one for the (co-)variances).

.. teaser-end


.. example

``keras-adf`` layers can be used exactly like the corresponding `Keras <https://keras.io/>`_
layers within a Keras model. However, as mentioned above, ADF layers take two inputs and produce two outputs
instead of one, so it is not possible to simply mix ADF and standard layers within the same model.

.. code-block:: python

    from tensorflow.keras import Input, Model
    from kerasadf.layers import Dense

    in_mean = Input((10,))
    in_var = Input((10,))
    out_mean, out_var  = Dense(10, activation="relu")([in_mean, in_var])
    model = Model([in_mean, in_var], [out_mean, out_var])

The `Overview <https://keras-adf.readthedocs.io/en/latest/overview.html>`_ and
`Examples <https://keras-adf.readthedocs.io/en/latest/examples.html>`_ sections
of our documentation provide more realistic and complete examples.

.. project-info-start

Project Information
===================

``keras-adf`` is released under the `MIT license <https://github.com/jmaces/keras-adf/blob/master/LICENSE>`_,
its documentation lives at `Read the Docs <https://keras-adf.readthedocs.io/en/latest/>`_,
the code on `GitHub <https://github.com/jmaces/keras-adf>`_,
and the latest release can be found on `PyPI <https://pypi.org/project/keras-adf/>`_.
It’s tested on Python 2.7 and 3.4+.

If you'd like to contribute to ``keras-adf`` you're most welcome.
We have written a `short guide <https://github.com/jmaces/keras-adf/blob/master/.github/CONTRIBUTING.rst>`_ to help you get you started!

.. project-info-end


.. literature-start

Further Reading
===============

Additional information on the algorithmic aspects of ``keras-adf`` can be found
in the following works:


- Jochen Gast, Stefan Roth,
  "Lightweight Probabilistic Deep Networks",
  2018
- Jan Macdonald, Stephan Wäldchen, Sascha Hauch, Gitta Kutyniok,
  "A Rate-Distortion Framework for Explaining Neural Network Decisions",
  2019

.. literature-end


Acknowledgments
===============

During the setup of this project we were heavily influenced and inspired by
the works of `Hynek Schlawack <https://hynek.me/>`_ and in particular his
`attrs <https://www.attrs.org/en/stable/>`_ package and blog posts on
`testing and packaing <https://hynek.me/articles/testing-packaging/>`_
and `deploying to PyPI <https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/>`_.
Thank you for sharing your experiences and insights.

Glossary
========

.. glossary::
    :sorted:

    probabilistic networks
        Probabilistic networks refer to any type of neural network involving
        probabilistic quantities (random variables). Typically these arise in
        context of Bayesian inference (Bayesian neural networks). The
        randomness can be used to model uncertainties in the input data, or
        also the model parameters. Applications include uncertainty quantification
        for neural networks and explaining or interpreting network predictions.

    assumed density filtering
        Assumed Density Filtering (ADF) is a concept from Bayesian inference related
        to Expectation Propagation (EP). Expectation propagation tries to iteratively
        approximate intractable multivariate probability distribution by distributions that
        factorize into several  simpler parts. The approximation is done with respect to
        Kullback-Leibler divergence. In the case of assuming that the simpler distributions
        are Gaussians this comes down to moment matching (matching the mean and covariance of
        the intractable distribution). ADF basically does only one step of the EP iteration, finding
        the best approximation for each of the factors in a chosen order just once. This means
        that approximations of the factors found first can not depend on the approximations
        of factors found later (unlike for EP, where several iterations of recalculating the
        approximations can capture correlations across factors). In our setting of neural networks,
        we chose the factors and their order according to the layers of the network and
        for simplicity assume Gaussian distributions for each layer.

    moments of a probability distribution
        The moments of a probability distribution or a random variable distributed
        according to that distributions are the expected values of the powers of that
        random variable. Central moments are the expectation values of the powers of
        the random variables shifted by its mean. We usually consider only central
        moments even though we not always explicitly state it. For example the first
        moment is just the expectation value, the second moment is variance, the third
        moment is skewness, and so on. For multivariate random vectors we also
        refer to the mean vector and covariances matrix as the first and second moments.

    moment matching
        Moment matching of distributions refers to the process of finding the parameters
        in a family of parametrized probability distributions that match the first few moments
        of a given distribution. The given distribution is typically not in the same family.
        This is for example done to approximate an intractable distribution by a simpler
        distribution from a family of tractable distributions. Approximating any distribution by
        a Gaussian distribution with respect to Kullback-Leibler divergence boils down to matching
        the first two moments (mean and covariance).

"""Custom strategies for hypothesis testing. """
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np

from hypothesis import assume


# constants for various tests
ALL_ACTIVATIONS = ["linear", "relu"]
COVARIANCE_MODES = ["diag", "half", "full"]


# array comparison helpers robust to precision loss
def assert_eq(x, y, atol=None, rtol=1e-7):
    """Robustly and symmetrically assert x == y componentwise. """
    if atol is None:
        atol = max(np.finfo(x.dtype).eps, np.finfo(y.dtype).eps)
    tol = atol + rtol * np.maximum(np.abs(x), np.abs(y), dtype=np.float64)
    np.testing.assert_array_less(np.abs(x - y), tol)


def assert_leq(x, y, atol=None, rtol=1e-7):
    """Robustly assert x <= y componentwise. """
    if atol is None:
        atol = max(np.finfo(x.dtype).eps, np.finfo(y.dtype).eps)
    mask = np.greater(x, y)
    np.testing.assert_allclose(x[mask], y[mask], atol=atol, rtol=rtol)


# data generation strategies
def clean_floats(min_value=-1e15, max_value=1e15, width=32):
    """Custom floating point number strategy.

    Working with very large or very small floats leads to over-/underflow
    problems. To avoid this we assume ``reasonable`` numbers for our tests.
    We exclude NaN, infinity, and negative infinity.

    The following ranges are recommended, so that squares (e.g. for variances)
    stay within the data type limits:
    -1e30 to +1e30 for 64-bit floats.
    -1e15  to +1e15  for 32-bit floats. (default)
    -200   to +200   for 16-bit floats.

    If your code really runs into floats outside this range probably something
    is wrong somewhere else.
    """
    if width == 64:
        min_value, max_value = np.clip(
            (min_value, max_value), -1e30, 1e30
        ).astype(np.float64)
    elif width == 32:
        min_value, max_value = np.clip(
            (min_value, max_value), -1e15, 1e15
        ).astype(np.float32)
    elif width == 16:
        min_value, max_value = np.clip(
            (min_value, max_value), -150, 150
        ).astype(np.float16)
    else:
        raise ValueError(
            "Invalid width parameted, expected 16, 32, or 64"
            "but got {}.".format(width)
        )
    return st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=min_value,
        max_value=max_value,
        width=width,
    )


@st.composite
def batched_float_array(
    draw, min_batch_size=None, min_data_dims=1, min_data_size=None
):
    """Float array strategy for different covariance modes.

    Generates tuples of batched mean and covariance arrays of shapes consistent
    for one of the allowed convariance modes {"diag", "half", "full"}.
    Shapes are arbitrary with at least two dimensions (first for batch_size,
    the remaining for the true data dimensions).
    Content can be any floating point data type.

    A minimum size for the batch dimension, the number of data dimensions and
    the product of all data dimensions can be specified respectively.

    Yields tuples (means, covariacnes, mode).
    """
    mode = draw(st.sampled_from(COVARIANCE_MODES))
    dtype = draw(hnp.floating_dtypes())
    bytes = dtype.itemsize
    bits = 8 * bytes
    shape = draw(hnp.array_shapes(min_dims=1 + min_data_dims))
    if min_batch_size is not None:
        assume(shape[0] >= min_batch_size)
    if min_data_size is not None:
        assume(np.prod(shape[1:]) >= min_data_size)
    means_ar = hnp.arrays(
        dtype,
        shape,
        elements=clean_floats(width=bits),
        fill=clean_floats(width=bits),
    )
    if mode == "diag":
        covariances_ar = hnp.arrays(
            dtype,
            shape,
            elements=clean_floats(width=bits, min_value=0.0),
            fill=clean_floats(width=bits, min_value=0.0),
        )
    elif mode == "half":
        rank = draw(st.integers(min_value=1, max_value=np.prod(shape[1:])))
        covariances_ar = hnp.arrays(
            dtype,
            (shape[0], rank) + shape[1:],
            elements=clean_floats(width=bits),
            fill=clean_floats(width=bits),
        )
    elif mode == "full":

        def fac_to_cov(L):
            shape = L.shape
            L = np.reshape(L, [shape[0], np.prod(shape[1:]), 1])
            return np.reshape(
                np.matmul(L, np.transpose(L, [0, 2, 1])),
                (shape[0],) + shape[1:] + shape[1:],
            )

        covariances_ar = hnp.arrays(
            dtype,
            shape,
            elements=clean_floats(width=bits),
            fill=clean_floats(width=bits),
        ).map(fac_to_cov)
    return draw(means_ar), draw(covariances_ar), mode

"""keras-adf layers API."""
from tensorflow.keras.layers import deserialize as _kerasdeserialize
from tensorflow.keras.layers import serialize

from .convolutional import Conv1D, Conv2D
from .core import Dense, Flatten
from .pooling import AveragePooling1D, AveragePooling2D


_LAYER_DICT = {
    "Conv1D": Conv1D,
    "Conv2D": Conv2D,
    "Dense": Dense,
    "Flatten": Flatten,
    "AveragePooling1D": AveragePooling1D,
    "AveragePooling2D": AveragePooling2D,
}


# overwrite keras layers by respective adf versions for deserialization
def deserialize(config, custom_objects=None):
    custom_objects = custom_objects or {}
    custom_objects.update(_LAYER_DICT)
    return _kerasdeserialize(config, custom_objects)


# export API
__all__ = list(_LAYER_DICT.keys()) + [
    "serialize",
    "deserialize",
]

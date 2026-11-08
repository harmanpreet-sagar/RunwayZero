"""Load Amelia's semantic_graph.pkl WITHOUT running arbitrary code.

Why this exists: a normal `pickle.load()` will call ANY function the file names, so a
downloaded pickle can run code on your machine. Amelia's map pickle only needs numpy to
rebuild its arrays (plus some networkx graph objects we don't use), so this loader:
  * allows only the numpy helpers that rebuild arrays,
  * swaps every networkx class for an inert placeholder,
  * refuses everything else (raises UnpicklingError).
"""
import io
import pickle

import numpy as np


class _Placeholder:
    """Stands in for networkx classes. Accepts any args/state and does nothing with them."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _frombuffer(buf, dtype, shape, order):
    """Rebuild a numpy array from raw bytes (what numpy's own pickling does)."""
    return np.frombuffer(buf, dtype=dtype).reshape(shape, order=order).copy()


def _scalar(dtype, data=None):
    """Rebuild a numpy scalar (e.g. np.float64(3.0)) from raw bytes."""
    return np.frombuffer(data, dtype=dtype)[0] if data is not None else dtype.type()


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # numpy moved these between numpy.core and numpy._core across versions; accept both.
        if module.startswith("numpy") and name == "_frombuffer":
            return _frombuffer
        if module.startswith("numpy") and name == "scalar":
            return _scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module.startswith("networkx"):
            return _Placeholder
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load(path):
    """Safely load a pickle file that only contains numpy data (+ ignorable networkx objects)."""
    with open(path, "rb") as fh:
        return SafeUnpickler(io.BytesIO(fh.read())).load()

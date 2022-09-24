import numpy as np
import importlib

# ==================================================================================================

def is_iterable(obj, length = None):
  """Checks if an object is an iterable of the (optional) given length, not including strings."""
  return hasattr(obj, "__iter__") and not isinstance(obj, str) and (len(obj) == length or length is None)

# ==================================================================================================

def is_number(obj):
  """Checks if an object is a scalar number (integer or float)."""
  return np.issubdtype(type(obj), np.number)

# ==================================================================================================

def is_array(obj, d = None):
  """Checks if an object is a d-dimensional NumPy array."""
  return isinstance(obj, np.ndarray) and (obj.ndim == d or d is None)

# ==================================================================================================

def try_import(module):
  """Returns the requested module if available, or else None."""
  try:
    return importlib.import_module(module)
  except ModuleNotFoundError:
    return None

# ==================================================================================================
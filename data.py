# Standard library imports.
from numbers import Number

# External library imports.
import scipy.sparse as sparse
import numpy as np

# Internal imports.
import gm2util.help as help
from gm2util.plot import Plot

# ==================================================================================================

class Data:

  # TODO: accept labels and units for x and y, for automatic plotting of self
  # TODO: maybe whole Quantity object which defines string labels, units, ref value, limits, etc.
  def __init__(self, x, y, cov = None, err = None):

    if len(x) != len(y):
      raise ValueError("Non-matching lengths for 'x' and 'y'.")
    if (cov is not None) and (err is not None):
      raise ValueError("Conflicting specification of both 'cov' and 'err'.")

    self.x = x.copy()
    self.y = y.copy()
    self.length = len(self.x)

    if err is not None:
      self.cov = sparse.diags(err**2)
    elif cov is not None:
      self.cov = cov.copy()
    else:
      self.cov = sparse.csr_matrix((self.length, self.length))

# ==================================================================================================

  def copy(self):
    return Data(self.x, self.y, self.cov)

# ==================================================================================================

  def _ensure_compatibility(self, other_val, other_cov, cross_cov):

    # If the other value is a numeric scalar...
    if isinstance(other_val, Number):

      # Ensure 'other_cov' is also a numeric scalar.
      if other_cov is None:
        other_cov = 0
      elif not isinstance(other_cov, Number):
        raise ValueError()
      
      # Ensure 'cross_cov' is a compatible 1-d array which can be broadcasted to a 2-d matrix.
      if cross_cov is None:
        cross_cov = 0
      elif other_cov == 0:
        # If 'other_cov' is 0, then 'cross_cov' doesn't make sense.
        raise ValueError()
      elif isinstance(cross_cov, np.ndarray) and cross_cov.shape == (len(self.x),):
        # Add a new axis so that 'cross_cov' broadcasts as 2-d matrix with first column repeated.
        cross_cov = cross_cov[:, None]
      else:
        raise ValueError()
      
    # If the other value is a Data object whose shape matches this one...
    elif isinstance(other_val, Data) and np.all(other_val.x == self.x):

      # Take 'other_val' and 'other_cov' from Data object.
      other_val, other_cov = other_val.y, other_val.cov

      # Ensure 'cross_cov' is a compatible 2-d matrix.
      if cross_cov is None:
        cross_cov = 0
      elif np.all(other_cov.diagonal() == 0):
        # If other values' variances are all 0, then 'cross_cov' doesn't make sense.
        raise ValueError()
      elif not (isinstance(cross_cov, np.ndarray) and cross_cov.shape == self.cov.shape):
        raise ValueError()
      
    else:
      raise ValueError()
    
    return other_val, other_cov, cross_cov

# ==================================================================================================

  # In-place negation of y-values.
  def negate(self):
    self.y = -1 * self.y
    return self

# ==================================================================================================

  # Override the (unary) - operator, returning a new Data object.
  def __neg__(self):
    return self.copy().negate()

# ==================================================================================================

  # In-place addition of y-values with another compatible Data object.
  def add(self, other_val, other_cov = None, cross_cov = None):
    other_val, other_cov, cross_cov = self._ensure_compatibility(other_val, other_cov, cross_cov)
    self.y = self.y + other_val
    self.cov = self.cov + cross_cov + np.transpose(cross_cov) + other_cov
    return self

# ==================================================================================================

  # Override the += operator.
  def __iadd__(self, other):
    return self.add(other)

# ==================================================================================================

  # Override the + operator, returning a new Data object.
  def __add__(self, other):
    return self.copy().add(other)

# ==================================================================================================

  # In-place subtraction of y-values with another compatible Data object.
  def subtract(self, other_val, other_cov = None, cross_cov = None):
    return self.add(-other_val, other_cov, -cross_cov if cross_cov is not None else None)

# ==================================================================================================

  # Override the -= operator.
  def __isub__(self, other):
    return self.subtract(other)

# ==================================================================================================

  # Override the - operator, returning a new Data object.
  def __sub__(self, other):
    return self.copy().subtract(other)

# ==================================================================================================

  # Helper for frequent pattern: np.outer(a, b) * cov_matrix, handling sparse/non-sparse cases.
  # If 'v' is a single array, use outer product with self, else if a tuple of 2 arrays, use those.
  @staticmethod
  def _outer_times_cov(v, cov):
    a, b = (v if help.is_iterable(v, 2) else v, v)
    if np.all(cov == 0):
      # Shortcut in case covariance matrix is 0.
      return 0
    else:
      return cov * (np.outer(a, b) if not sparse.issparse(cov) else sparse.diags(a * b))

# ==================================================================================================

  # In-place multiplication of y-values with another compatible Data object.
  def multiply(self, other_val, other_cov = None, cross_cov = None):
    other_val, other_cov, cross_cov = self._ensure_compatibility(other_val, other_cov, cross_cov)
    cross_term = Data._outer_times_cov((self.y, other_val), cross_cov)
    self.cov = Data._outer_times_cov(other_val, self.cov) + cross_term + np.transpose(cross_term) + Data._outer_times_cov(other_val, other_cov)
    self.y = self.y * other_val
    return self

# ==================================================================================================

  # Override the *= operator.
  def __imul__(self, other):
    return self.multiply(other)

# ==================================================================================================

  # Override the * operator, returning a new Data object.
  def __mul__(self, other):
    return self.copy().multiply(other)

# ==================================================================================================

  # In-place division of y-values with another compatible Data object.
  def divide(self, other_val, other_cov = None, cross_cov = None):
    # TODO: what to do about division by zero...
    other_val, other_cov, cross_cov = self._ensure_compatibility(other_val, other_cov, cross_cov)
    cross_term = Data._outer_times_cov((1 / other_val, self.y / other_val**2), cross_cov)
    self.cov = Data._outer_times_cov(1 / other_val, self.cov) - cross_term - np.transpose(cross_term) + Data._outer_times_cov(self.y / other_val**2, other_cov)
    self.y = self.y / other_val
    return self

# ==================================================================================================

  # Override the /= operator.
  def __itruediv__(self, other):
    return self.divide(other)

# ==================================================================================================

  # Override the / operator, returning a new Data object.
  def __truediv__(self, other):
    return self.copy().divide(other)

# ==================================================================================================

  # In-place exponentiation of y-values with a scalar constant.
  def power(self, n):
    self.cov = Data._outer_times_cov(self.y**(n-1), self.cov)
    self.y = self.y ** n
    return self

# ==================================================================================================

  # Override the **= operator.
  def __ipow__(self, n):
    return self.power(n)

# ==================================================================================================

  # Override the ** operator, returning a new Data object.
  def __pow__(self, n):
    return self.copy().power(n)

# ==================================================================================================

  # Compute the error vector from the diagonal of the covariance matrix. Return None if all zero.
  def err(self):
    errors = np.sqrt(self.cov.diagonal())
    return errors if np.any(errors != 0) else None

# ==================================================================================================

  # Remap x-values according to the given function, which must be monotonic.
  def map(self, function):
    new_x = function(self.x)
    new_dx = np.diff(new_x)
    if np.all(new_dx > 0):
      self.x = new_x
    elif np.all(new_dx < 0):
      self.x = np.flip(new_x)
      self.y = np.flip(self.y)
      self.cov = np.flip(self.cov) if not sparse.issparse(self.cov) else sparse.diags(np.flip(self.cov.diagonal()))
    else:
      raise ValueError("Remapped x-values must be monotonic.")
    return self

# ==================================================================================================

  # Integrate y(x) using Simpson's rule for quadratic interpolation.
  # SciPy offers this, but we need internal details to propagate errors, so reimplement here.
  def integrate(self, error = False):

    dx = np.diff(self.x) # Interval widths.
    w = np.zeros(len(self.x)) # Weights for each y-value in the linear combination.

    # Integrate parabolic interpolation of 3 points (start, mid, end).
    # Then step by 2 so that end -> start for next group.
    # Keep a running total of the weight each y-value contributes.
    for i in range(0, len(w) - 2, 2):
      c = (dx[i] + dx[i+1]) / 6
      w[i] += c * (2 - dx[i+1] / dx[i])
      w[i+1] += c * (dx[i] + dx[i+1])**2 / (dx[i] * dx[i+1])
      w[i+2] += c * (2 - dx[i] / dx[i+1])

    # If there's an even number of points, treat the last interval using a trapezoid rule.
    if len(w) % 2 == 0:
      w[-2] += dx[-1] / 2
      w[-1] += dx[-1] / 2

    # Return the weighted linear combination of the y-values, optionally with error propagation.
    result = w @ self.y
    return result if not error else (result, np.sqrt(w @ (self.cov @ w)))

# ==================================================================================================

  def mean(self, weights = None, error = False):

    if weights is None:
      weights = np.ones(len(self.x))

    total = np.sum(weights)
    mean = (weights @ self.y) / total

    if not error:
      return mean
    else:
      mean_err = np.sqrt(weights @ (self.cov @ weights)) / total
      return mean, mean_err

# ==================================================================================================

  def std(self, weights = None, error = False):

    if weights is None:
      weights = np.ones(len(self.x))

    total = np.sum(weights)
    mean = self.mean(weights, error = False)
    var = weights @ (self.y - mean)**2 / total
    std = np.sqrt(var)

    if not error:
      return std
    else:
      temp = weights * ((self.y - mean)**2 - 2 * mean * (self.y - mean) - var)
      var_err = np.sqrt(temp @ (self.cov @ temp))
      std_err = var_err / (2 * std)
      return std, std_err

# ==================================================================================================

  # TODO: spline interpolation, optional order, optional smoothing.
  # store spline object from original data, re-use if already exists.

# ==================================================================================================

  def plot(self, plot = None, stats = False, **kwargs):

    if plot is None:
      plot = Plot()

    plot.plot(self, **kwargs)
    if stats:
      mean, mean_err = self.mean(error = True)
      std, std_err = self.std(error = True)
      plot.draw_horizontal(mean)
      plot.horizontal_spread(std, mean)
      # plot.databox(
      #   ("mean", mean, mean_err),
      #   ("std", std, std_err)
      # )

    return plot

# ==================================================================================================

  # Return a dictionary of the x, y, and cov data, with optional label as a prefix on the keys.
  def _data_dict(self, label = None):
    prefix = "" if label is None else f"{label}/"
    return {
      f"{prefix}x": self.x,
      f"{prefix}y": self.y,
      f"{prefix}cov": self.cov # TODO: will this work with sparse cov?
    }

# ==================================================================================================

  # Save the Data object to disk in NumPy format.
  def save(self, filename, label = None):
    np.savez(filename, **self._data_dict(label))

# ==================================================================================================

  # Load a Data object saved to disk in NumPy format.
  @staticmethod
  def load(filename, label = None):
    data = np.load(filename, allow_pickle = True)
    prefix = "" if label is None else f"{label}/"
    x = data[f"{prefix}x"]
    y = data[f"{prefix}y"]
    cov = data[f"{prefix}cov"]
    return Data(x, y, cov)

# ==================================================================================================

if __name__ == "__main__":

  x = np.arange(100)
  data1 = Data(x, x)
  data2 = Data(x, x**2)

  add_test = data1 + data2
  print(repr(add_test.cov))

  sub_test = data1 + data2
  print(repr(sub_test.cov))

  mul_test = data1 * data2
  print(repr(mul_test.cov))

  div_test = data1 / data2
  print(repr(div_test.cov))

  pow_test = data1 ** 3
  print(repr(pow_test.cov))
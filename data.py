from tkinter import Label
import scipy.sparse as sparse
import numpy as np

import gm2util.pyutil as pyutil
#root = pyutil.try_import("ROOT")

# ======================================================================================================================

class Data:

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

# ======================================================================================================================

  def copy(self):
    return Data(self.x, self.y, self.cov)

# ======================================================================================================================

  def _ensure_compatibility(self, other, cross_cov = None):
    if not isinstance(other, Data):
      raise ValueError(f"Cannot add object of type {type(other)} to Data object.")
    if np.any(self.x != other.x):
      raise ValueError("Data objects' x-values do not match.")
    if cross_cov is not None and cross_cov.shape != (self.length, self.length):
      raise ValueError("Cross-covariance of Data objects inconsistent with Data shapes.")

# ======================================================================================================================

  # In-place negation of y-values.
  def negate(self):
    self.y *= -1
    return self

# ======================================================================================================================

  # Override the (unary) - operator, returning a new Data object.
  def __neg__(self):
    return self.copy().negate()

# ======================================================================================================================

  # In-place addition of y-values with another compatible Data object.
  def add(self, other, cross_cov = None):
    self._ensure_compatibility(other, cross_cov)
    self.y = self.y + other.y
    self.cov = self.cov + other.cov
    if cross_cov is not None:
      self.cov = self.cov + 2 * cross_cov
    return self

# ======================================================================================================================

  # Override the += operator.
  def __iadd__(self, other):
    return self.add(other)

# ======================================================================================================================

  # Override the + operator, returning a new Data object.
  def __add__(self, other):
    return self.copy().add(other)

# ======================================================================================================================

  # In-place subtraction of y-values with another compatible Data object.
  def subtract(self, other, cross_cov = None):
    return self.add(-other, -cross_cov if cross_cov is not None else None)

# ======================================================================================================================

  # Override the -= operator.
  def __isub__(self, other):
    return self.subtract(other)

# ======================================================================================================================

  # Override the - operator, returning a new Data object.
  def __sub__(self, other):
    return self.copy().subtract(other)

# ======================================================================================================================

  # Helper function for frequently-used pattern: np.outer(a, b) * cov_matrix, handling sparse/non-sparse cases.
  # If argument 'v' is a single array, use outer product with self, else if a tuple of 2 arrays, use those two.
  @staticmethod
  def _outer_times_cov(v, cov):
    a, b = (v if pyutil.is_iterable(v, 2) else v, v)
    return cov * (np.outer(a, b) if not sparse.issparse(cov) else sparse.diags(a * b))

# ======================================================================================================================

  # In-place multiplication of y-values with another compatible Data object.
  def multiply(self, other, cross_cov = None):
    self._ensure_compatibility(other, cross_cov)
    self.cov = Data._outer_times_cov(other.y, self.cov) + Data._outer_times_cov(self.y, other.cov)
    if cross_cov is not None:
      temp = Data._outer_times_cov((self.y, other.y), cross_cov)
      self.cov = self.cov + (temp + temp.T)
    self.y = self.y * other.y
    return self

# ======================================================================================================================

  # Override the *= operator.
  def __imul__(self, other):
    return self.multiply(other)

# ======================================================================================================================

  # Override the * operator, returning a new Data object.
  def __mul__(self, other):
    return self.copy().multiply(other)

# ======================================================================================================================

  # In-place division of y-values with another compatible Data object.
  def divide(self, other, cross_cov = None):
    # TODO: what to do about division by zero...
    self._ensure_compatibility(other, cross_cov)
    self.cov = Data._outer_times_cov(1 / other.y, self.cov) + Data._outer_times_cov(self.y / other.y**2, other.cov)
    if cross_cov is not None:
      temp = Data._outer_times_cov((1 / other.y, self.y / other.y**2), cross_cov)
      self.cov = self.cov - (temp + temp.T)
    self.y = self.y / other.y
    return self

# ======================================================================================================================

  # Override the /= operator.
  def __itruediv__(self, other):
    return self.divide(other)

# ======================================================================================================================

  # Override the / operator, returning a new Data object.
  def __truediv__(self, other):
    return self.copy().divide(other)

# ======================================================================================================================

  # In-place exponentiation of y-values with a scalar constant.
  def power(self, n):
    self.cov = Data._outer_times_cov(self.y**(n-1), self.cov)
    self.y = self.y ** n
    return self

# ======================================================================================================================

  # Override the **= operator.
  def __ipow__(self, n):
    return self.power(n)

# ======================================================================================================================

  # Override the ** operator, returning a new Data object.
  def __pow__(self, n):
    return self.copy().power(n)

# ======================================================================================================================

  # Compute the error vector from the diagonal of the covariance matrix. Return None if all zero.
  def err(self):
    errors = np.sqrt(self.cov.diagonal())
    return errors if np.any(errors != 0) else None

# ======================================================================================================================

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

# ======================================================================================================================

  # Integrate y(x) using Simpson's rule for quadratic interpolation.
  # SciPy offers this algorithm, but we need internal details in order to propagate errors, so it is reimplemented here.
  def integrate(self, error = False):

    dx = np.diff(self.x) # Interval widths.
    w = np.zeros(len(self.x)) # Weights for each y-value in the linear combination for Simpson's rule.

    # Integrate parabolic interpolation of 3 points (start, mid, end), then step by 2 so that end -> start for next group.
    # Keep a running total of the weight each y-value contributes, since start/end will enter multiple groups.
    for i in range(0, len(w) - 2, 2):
      c = (dx[i] + dx[i+1]) / 6
      w[i] += c * (2 - dx[i+1] / dx[i]) # 'start' point contributes y[i] * w[i] to area.
      w[i+1] += c * (dx[i] + dx[i+1])**2 / (dx[i] * dx[i+1]) # 'mid' point contributes y[i+1] * w[i+1] to area.
      w[i+2] += c * (2 - dx[i] / dx[i+1]) # 'end' point contributes y[i+2] * w[i+2] to area.

    # If there's an even number of points, treat the last incomplete interval using a trapezoid rule.
    if len(w) % 2 == 0:
      w[-2] += dx[-1] / 2
      w[-1] += dx[-1] / 2

    # Return the weighted linear combination of the y-values, optionally with error propagation.
    result = w @ self.y
    return result if not error else (result, np.sqrt(w @ (self.cov @ w)))

# ======================================================================================================================

  def mean(self, weights = None, error = False):

    if weights is None:
      weights = np.ones(len(self.x))

    total = np.sum(weights)
    mean = (weights @ self.y) / total

    return mean if not error else (mean, np.sqrt(weights @ (self.cov @ weights)) / total)

# ======================================================================================================================

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

# ======================================================================================================================

  # TODO: spline interpolation, optional order, optional smoothing. store spline obj from original data, re-use if already exists.

# ======================================================================================================================

  # TODO: plot

# ======================================================================================================================

  # Return a dictionary representation of the x, y, and covariance data, with optional label as a prefix on the keys.
  def _data_dict(self, label = None):
    prefix = "" if label is None else f"{label}/"
    return {
      f"{prefix}x": self.x,
      f"{prefix}y": self.y,
      f"{prefix}cov": self.cov # TODO: will this work with sparse cov?
    }

# ======================================================================================================================

  # Save the Data object to disk in NumPy format.
  def save(self, filename, label = None):
    np.savez(filename, **self._data_dict(label))

# ======================================================================================================================

  # Load a Data object saved to disk in NumPy format.
  @staticmethod
  def load(filename, label = None):
    data = np.load(filename, allow_pickle = True)
    prefix = "" if label is None else f"{label}/"
    x = data[f"{prefix}x"]
    y = data[f"{prefix}y"]
    cov = data[f"{prefix}cov"]
    return Data(x, y, cov)

# ======================================================================================================================

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
import scipy.sparse as sparse
import numpy as np

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
      self.cov = sparse.diags(np.zeros(len(self.x)))

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

  @staticmethod
  def _outer_times_cov(v, w, cov):
    return cov * (np.outer(v, w) if not sparse.issparse(cov) else sparse.diags(v * w))

# ======================================================================================================================

  # In-place multiplication of y-values with another compatible Data object.
  def multiply(self, other, cross_cov = None):
    self._ensure_compatibility(other, cross_cov)
    self.cov = Data._outer_times_cov(other.y, other.y, self.cov) + Data._outer_times_cov(self.y, self.y, other.cov)
    if cross_cov is not None:
      temp = Data._outer_times_cov(self.y, other.y, cross_cov)
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
    # TODO: what to do about division by zero... cov doesn't remain sparse with np.nan from division
    self._ensure_compatibility(other, cross_cov)
    self.cov = Data._outer_times_cov(1/other.y, 1/other.y, self.cov) + Data._outer_times_cov(self.y / other.y**2, self.y / other.y**2, other.cov)
    if cross_cov is not None:
      temp = Data._outer_times_cov(1 / other.y, self.y / other.y**2, cross_cov)
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
    self.cov = Data._outer_times_cov(self.y**(n-1), self.y**(n-1), self.cov)
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
    errors = np.sqrt(self.cov.diagonal()) if self.cov is not None else None
    if errors is not None and np.any(errors != 0):
      return errors
    else:
      return None

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

    dx = np.diff(self.x) # interval widths
    w = np.zeros(len(self.x)) # weights for each y-value in the linear combination for Simpson's rule

    # step through the data points by 2, and add the corresponding weight for each point
    for i in range(0, len(w) - 2, 2):
      a = (dx[i] + dx[i+1]) / 6
      w[i] += a * (2 - dx[i+1] / dx[i])
      w[i+1] += a * (dx[i] + dx[i+1])**2 / (dx[i] * dx[i+1])
      w[i+2] += a * (2 - dx[i] / dx[i+1])

    # if there's an even number of points, treat the last interval using a trapezoid rule
    if len(w) % 2 == 0:
      w[-2] += dx[-1] / 2
      w[-1] += dx[-1] / 2

    result = w @ self.y
    if not error:
      return result
    else:
      return result, np.sqrt(w @ (self.cov @ w))

# ======================================================================================================================

  # TODO: mean and std, optional weights, optionally propagating covariance to result

# ======================================================================================================================

  # TODO: spline interpolation, optional order, optional smoothing. store spline obj from original data, re-use if already exists.

# ======================================================================================================================

  # TODO: plot

# ======================================================================================================================

  # TODO: saving/loading using both NumPy and ROOT formats

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
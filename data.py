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
      self.cov = None

# ======================================================================================================================

  def copy(self):
    return Data(self.x, self.y, self.cov)

# ======================================================================================================================

  def _ensure_compatibility(self, other, cross_cov = None):
    if not isinstance(other, Data):
      raise ValueError(f"Cannot add object of type {type(other)} to Data object.")
    if self.x != other.x:
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
    self.y += other.y
    self.cov += other.cov
    if cross_cov is not None:
      self.cov += 2 * cross_cov
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

  # In-place multiplication of y-values with another compatible Data object.
  def multiply(self, other, cross_cov = None):
    self._ensure_compatibility(other, cross_cov)
    self.cov *= np.outer(other.y, other.y) if not sparse.issparse(self.cov) else other.y**2
    self.cov += (np.outer(self.y, self.y) if not sparse.issparse(other.cov) else self.y**2) * other.cov
    if cross_cov is not None:
      temp = (np.outer(self.y, other.y) if not sparse.issparse(cross_cov) else self.y * other.y) * cross_cov
      self.cov += temp + temp.T
    self.y *= other.y
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
    self._ensure_compatibility(other, cross_cov)
    self.cov /= np.outer(other.y, other.y) if not sparse.issparse(self.cov) else other.y**2
    self.cov += (np.outer(self.y, self.y) / np.outer(other.y, other.y)**2 if not sparse.issparse(other.cov) else self.y**2 / other.y**4) * other.cov
    if cross_cov is not None:
      temp = (np.outer(1 / other.y, self.y / other.y**2) if not sparse.issparse(cross_cov) else self.y / other.y**3) * cross_cov
      self.cov -= temp + temp.T
    self.y /= other.y
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
    self.cov *= n**2 * (np.outer(self.y, self.y) if not sparse.issparse(self.cov) else self.y**2)**(n - 1)
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
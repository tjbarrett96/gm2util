import scipy.sparse as sparse
import numpy as np

# ======================================================================================================================

class Data:

  def __init__(self, x, y, cov = None, err = None):

    if (cov is not None) and (err is not None):
      raise ValueError("Conflicting specification of both 'cov' and 'err'.")

    self.x = x.copy()
    self.y = y.copy()

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
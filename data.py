import scipy.sparse as sparse
import numpy as np

class Data:

  def __init__(self, x, y, cov = None, err = None):

    if cov is not None and err is not None:
      raise ValueError("Conflicting specification of both 'cov' and 'err'.")

    self.x = x
    self.y = y

    if err is not None:
      self.cov = sparse.diags(err**2)
      self.inv_cov = sparse.diags(1 / err**2)
    elif cov is not None:
      self.cov = cov
      self.inv_cov = np.linalg.inv(cov)
    else:
      self.cov = None
      self.inv_cov = None
import sympy as sp
from sympy.utilities.lambdify import implemented_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from data import Data

import time
import re

# ======================================================================================================================

class Fit:

  def __init__(self, data, expr, arg = "x", definitions = None):

    # The Data object to fit, which wraps the numerical data as data.x, data.y, and data.cov.
    self.data = data

    # Attach custom numerical definitions (lambda expressions) to custom functions named in the 'expr' string.
    if definitions is not None:
      custom_functions = {name: implemented_function(name, lambda x: impl) for name, impl in definitions.items()}
    else:
      custom_functions = None

    # Parse the model function and independent variable strings into SymPy expressions.
    self.sp_expr = sp.parse_expr(expr, local_dict = custom_functions)
    self.sp_arg = sp.Symbol(arg)

    # Identify the fit parameters as the set of all unbound symbols, minus the independent variable.
    # Convert the set into an ordered list, and sort by order of appearance within the user's function string.
    self.sp_params = sorted(
      list(self.sp_expr.free_symbols - {self.sp_arg}),
      key = lambda p: re.search(rf"\b{p.name}\b", expr).start()
    )

    # Compute the jacobian vector of first derivatives with respect to each parameter.
    self.sp_jac = [sp.diff(self.sp_expr, p) for p in self.sp_params]

    # Compute the hessian matrix of second derivatives with respect to each pair of parameters.
    self.sp_hess = [[sp.diff(self.sp_expr, p, q) for p in self.sp_params] for q in self.sp_params]

    # Convert the SymPy model function, jacobian, and hessian into NumPy functions.
    temp_np_expr = sp.lambdify([self.sp_arg, *self.sp_params], self.sp_expr)
    temp_np_jac = [sp.lambdify([self.sp_arg, *self.sp_params], df_dp) for df_dp in self.sp_jac]
    temp_np_hess = [[sp.lambdify([self.sp_arg, *self.sp_params], df_dpdq) for df_dpdq in row] for row in self.sp_hess]

    # If the sp.lambdify function is independent of x, then it returns a scalar, which doesn't match the shape of x.
    # Wrap each lambda with another lambda to ensure the result shape matches len(x) in all cases.
    self.np_expr = lambda x, *p: temp_np_expr(x, *p) * np.ones(len(x))
    self.np_jac = lambda x, *p: np.array([df_dp(x, *p) * np.ones(len(x)) for df_dp in temp_np_jac])
    self.np_hess = lambda x, *p: np.array([[df_dpdq(x, *p) * np.ones(len(x)) for df_dpdq in row] for row in temp_np_hess])
    # TODO: get the computations involving these objects working elegantly with broadcasting

# ======================================================================================================================

  def fit(self, guess = None, hopping = False):

    self.guess = guess if guess is not None else np.ones(len(self.sp_params))

    # Minimize the chi-squared using BFGS, repeated with random steps in the initial conditions ("basin hopping").
    if hopping:

      self.opt_result = opt.basinhopping(
        self.chi2,
        x0 = self.guess,
        minimizer_kwargs = {
          "method": "BFGS",
          "jac": self.chi2_jac
        }
      ).lowest_optimization_result

    else:

      # how to use basinhopping intelligently to keep things fast? single local minimize is ~100x faster...
      # maybe try single minimization at given seed first, and proceed to basin hopping if result is poor? e.g. didn't converge, or chi2 too big?
      self.opt_result = opt.minimize(
        self.chi2,
        x0 = self.guess,
        method = "BFGS",
        jac = self.chi2_jac
      )
    
    # Extract the optimized parameters from the minimization result.
    self.p_opt = self.opt_result.x

    # Calculate the parameter covariance matrix, but only if the data had a covariance matrix -- otherwise not meaningful.
    if self.data.cov is not None:
      self.p_cov = np.linalg.inv(self.chi2_hess(self.p_opt)) # TODO: should there be a 1/2 here or not????
    else:
      self.p_cov = None

    # Calculate the minimized chi2 and chi2/ndf.
    self.min_chi2 = self.opt_result.fun
    self.ndf = len(self.data.y) - len(self.sp_params)
    self.chi2_ndf = self.min_chi2 / self.ndf

# ======================================================================================================================

  # Compute the chi-squared at the given vector of parameter values.
  def chi2(self, p):
    res = self.np_expr(self.data.x, *p) - self.data.y
    return (res * (self.data.inv_cov @ res)).sum()

# ======================================================================================================================

  # Compute the chi-squared jacobian vector (with respect to the parameters) at the given vector of parameter values.
  def chi2_jac(self, p):
    res = self.np_expr(self.data.x, *p) - self.data.y
    jac = self.np_jac(self.data.x, *p)
    return 2 * np.array([(jac[i] * (self.data.inv_cov @ res)).sum() for i in range(len(jac))])

# ======================================================================================================================

  # Compute the chi-squared hessian matrix (with respect to the parameters) at the given vector of parameter values.
  def chi2_hess(self, p):
    res = self.np_expr(self.data.x, *p) - self.data.y
    jac = self.np_jac(self.data.x, *p)
    hess = self.np_hess(self.data.x, *p)
    return 2 * np.array([
      [(hess[i][j] * (self.data.inv_cov @ res) + jac[i] * (self.data.inv_cov @ jac[j])).sum() for j in range(len(jac))]
      for i in range(len(jac))
    ])
  
# ======================================================================================================================

  # Evaluate the model NumPy function using the optimized parameters.
  def __call__(self, x):
    return self.np_expr(x, *self.p_opt)

# ======================================================================================================================

  # Calculate the covariance matrix of the optimal fitted curve evaluated at the given x values.
  def cov(self, x):
    if self.p_cov is not None:
      jac = self.np_jac(x, *self.p_opt)
      return jac.T @ self.p_cov @ jac
    else:
      return None

# ======================================================================================================================

  # Calculate the one-sigma error band of the optimal fitted curve evaluated at the given x values.
  def err(self, x):
    cov = self.cov(x)
    if cov is not None:
      return np.sqrt(np.diag(cov))

# ======================================================================================================================

if __name__ == "__main__":

  std = 2

  x = np.linspace(0, 10, 1000)
  y = (10 * np.cos(3*x) + x**2) + np.random.normal(0, std, size = len(x))
  err = np.ones(len(x)) * std
  data = Data(x, y, err = err)

  fit = Fit(data, "a * cos(b*x) + c * f(x)", definitions = {"f": x**2})
  fit.fit(hopping = True)
  print(fit.p_opt)
  print(np.sqrt(np.diag(fit.p_cov)))
  print(fit.chi2_ndf)

  fit_result = fit(x)
  fit_err = fit.err(x)

  plt.plot(x, y)
  plt.plot(x, fit_result)
  plt.fill_between(x, fit_result - fit_err, fit_result + fit_err)
  plt.show()
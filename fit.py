import sympy as sp
from sympy.utilities.lambdify import implemented_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats

from gm2pal.data import Data
from gm2pal.plot import Plot
import gm2pal.io as io

import time
import re

# ======================================================================================================================

class Fit:

  def __init__(self, data, expr, arg = "x", definitions = None):

    # The Data object to fit, which wraps the numerical data as data.x, data.y, and data.cov.
    self.data = data

    # Attach custom numerical definitions (lambda expressions) to custom functions named in the 'expr' string.
    if definitions is not None:
      custom_functions = {
        name: implemented_function(name, impl if callable(impl) else (lambda x: impl)) 
        for name, impl in definitions.items()
      }
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

# ======================================================================================================================

  def fit(self, guess = None, hopping = False):

    start_time = time.perf_counter()

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
      self.p_err = np.sqrt(np.diag(self.p_cov))
    else:
      self.p_cov = None
      self.p_err = [None] * len(self.p_opt)

    # Calculate the minimized chi2 and chi2/ndf.
    self.min_chi2 = self.opt_result.fun
    self.ndf = len(self.data.y) - len(self.sp_params)
    self.chi2_ndf = self.min_chi2 / self.ndf
    self.err_chi2_ndf = np.sqrt(2 / self.ndf) # std. dev. of reduced chi2 distribution

    self.duration = time.perf_counter() - start_time

# ======================================================================================================================

  # Compute the chi-squared at the given vector of parameter values.
  def chi2(self, p):
    # TODO: only take the nonlinear parameters here, and automatically determine the linear parameters here
    res = self.np_expr(self.data.x, *p) - self.data.y
    return np.matmul(res, self.data.inv_cov.dot(res))

# ======================================================================================================================

  # Compute the chi-squared jacobian vector (with respect to the parameters) at the given vector of parameter values.
  def chi2_jac(self, p):
    res = self.np_expr(self.data.x, *p) - self.data.y
    jac = self.np_jac(self.data.x, *p)
    return 2 * np.matmul(jac, self.data.inv_cov.dot(res))

# ======================================================================================================================

  # Compute the chi-squared hessian matrix (with respect to the parameters) at the given vector of parameter values.
  def chi2_hess(self, p):
    res = self.np_expr(self.data.x, *p) - self.data.y
    jac = self.np_jac(self.data.x, *p)
    hess = self.np_hess(self.data.x, *p)
    return 2 * (np.matmul(hess, self.data.inv_cov.dot(res)) + np.matmul(jac, self.data.inv_cov.dot(jac.T)))
  
# ======================================================================================================================

  # Calculate the two-sided p-value from the chi2 distribution with 'ndf' degrees of freedom.
  def pval(self):
    # The difference between this fit's chi2 and the mean of the distribution.
    mean_diff = abs(self.min_chi2 - self.ndf)
    # The total probability of drawing a chi2 sample farther from the mean on either the left or right side.
    return stats.chi2.cdf(self.ndf - mean_diff, self.ndf) + (1 - stats.chi2.cdf(self.ndf + mean_diff, self.ndf))

# ======================================================================================================================

  # Evaluate the model NumPy function using the optimized parameters.
  def __call__(self, x):
    return self.np_expr(x, *self.p_opt)

# ======================================================================================================================

  # Calculate the covariance matrix of the optimal fitted curve evaluated at the given x values.
  def cov(self, x):
    if self.p_cov is not None:
      jac = self.np_jac(x, *self.p_opt)
      return np.matmul(jac.T, np.matmul(self.p_cov, jac))

# ======================================================================================================================

  # Calculate the one-sigma error band of the optimal fitted curve evaluated at the given x values.
  def err(self, x):
    cov = self.cov(x)
    if cov is not None:
      return np.sqrt(np.diag(cov))

# ======================================================================================================================

  def print(self, quality = True, parameters = True):

    lines = []

    if parameters:
      lines += io.format_values(*[
        (p.name, p_opt, p_err)
        for p, p_opt, p_err in zip(self.sp_params, self.p_opt, self.p_err)
      ])

    if quality and self.data.cov is not None:
      lines += io.format_values(
        ("chi2/ndf", self.chi2_ndf, self.err_chi2_ndf),
        ("p-value", self.pval())
      )

    print(f"Fit completed in {self.duration:.{io.get_decimal_places(self.duration, 2)}f} seconds.")
    print(io.align(*lines, margin = 4))

# ======================================================================================================================

if __name__ == "__main__":

  std = 5

  x = np.linspace(0, 10, 200)
  y = (10 * np.cos(3*x) + x**2) + np.random.normal(0, std, size = len(x))
  err = np.ones(len(x)) * std
  data = Data(x, y, err = err)
  fit = Fit(data, "a * cos(b*x) + c * f(x)", definitions = {"f": x**2})
  fit.fit(hopping = True)
  fit.print()

  plot = Plot()
  plot.plot(x, y, err, line = None, label = "Data")
  plot.plot(x, fit(x), fit.err(x), error_mode = "band", label = "Fit")
  plot.databox(
    (r"$\chi^2$/ndf", fit.chi2_ndf, fit.err_chi2_ndf),
    ("$p$-value", fit.pval())
  )
  plot.labels(r"$\sigma$", "y", "Title")
  plot.save("test.pdf")
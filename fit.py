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

    # Attach numerical definitions to custom functions named in the 'expr' string.
    # May be callable single-variable functions, or discrete numeric vectors which must match the shape of the data.
    sp_local_dict = {}
    if definitions is not None:
      for name, impl in definitions.items():
        if callable(impl):
          sp_local_dict[name] = implemented_function(name, impl)
        elif len(impl) == len(self.data.x):
          sp_local_dict[name] = implemented_function(name, lambda x: impl)
        else:
          raise ValueError()

    # Parse the model function and independent variable strings into SymPy expressions.
    self.sp_expr = sp.parse_expr(expr, local_dict = sp_local_dict)
    self.sp_arg = sp.Symbol(arg)

    # Identify the fit parameters as the set of all unbound symbols, minus the independent variable.
    # Convert the set into an ordered list, and sort by order of appearance within the user's function string.
    self.sp_params = sorted(
      list(self.sp_expr.free_symbols - {self.sp_arg}),
      key = lambda p: re.search(rf"\b{p.name}\b", expr).start()
    )
    self.str_params = [p.name for p in self.sp_params]

    # Compute the jacobian vector of first derivatives with respect to each parameter.
    self.sp_jac = [sp.diff(self.sp_expr, p) for p in self.sp_params]

    # Compute the hessian matrix of second derivatives with respect to each pair of parameters.
    self.sp_hess = [[sp.diff(self.sp_expr, p, q) for p in self.sp_params] for q in self.sp_params]

    # Identify which parameters in the sorted parameter list are linear.
    self.where_linear = np.full(len(self.sp_params), False)
    self.where_linear[self._identify_linear_params()] = True
    self.where_nonlinear = ~self.where_linear

    # Separate the symbolic expression into a linear part and nonlinear part (with respect to the parameters).
    self.sp_expr_linear = 0
    for i, param in enumerate(self.sp_params):
      if self.where_linear[i]:
        self.sp_expr_linear += self.sp_jac[i] * param
    self.sp_expr_nonlinear = self.sp_expr - self.sp_expr_linear

    # Helper function for converting a nested list of SymPy expressions into a NumPy function with the same structure.
    def np_wrap(args, f):
      if isinstance(f, list):
        return lambda x, *p: np.array([np_wrap(args, item)(x, *p) for item in f])
      else:
        lambdified = sp.lambdify(args, f)
        # Ensure the function output matches the length of x. It won't by default if the function is independent of x.
        return lambda x, *p: lambdified(x, *p) * np.ones(len(x))

    # Convert SymPy expressions into NumPy functions.
    arg_list = [self.sp_arg, *self.sp_params]
    self.np_expr = np_wrap(arg_list, self.sp_expr)
    self.np_jac = np_wrap(arg_list, self.sp_jac)
    self.np_hess = np_wrap(arg_list, self.sp_hess)
    self.np_expr_nonlinear = np_wrap(
      [self.sp_arg] + [p for i, p in enumerate(self.sp_params) if self.where_nonlinear[i]],
      self.sp_expr_nonlinear
    )

    self.fixed = {}
    self.where_fixed = np.full(len(self.sp_params), False)
    self.where_float = np.full(len(self.sp_params), True)
    self.fixed_params = []
    self.float_params = []
    self.fixed_template = np.full(len(self.sp_params), np.nan)
    self._update_fixed()

# ======================================================================================================================

  # TODO: accept step sizes (setting scale for knowledge of initial seeds) for basin hopping
  def fit(self, guess = None, hopping = False):

    start_time = time.perf_counter()

    if guess is None:
      guess = {}
    if isinstance(guess, dict):
      seeds = np.ones(len(self.float_params))
      for i, param in enumerate(self.float_params):
        if param in guess:
          seeds[i] = guess[param]
    else:
      raise ValueError()

    # Minimize the chi-squared using BFGS, repeated with random steps in the initial conditions ("basin hopping").
    if hopping:

      self.opt_result = opt.basinhopping(
        self._eval_chi2,
        x0 = seeds,
        callback = lambda p, f, acc: (f/self.ndf <= 1 + self.err_chi2_ndf), # quit early if chi2/ndf within 1 sigma of mean
        disp = True,
        minimizer_kwargs = {
          "method": "BFGS",
          "jac": self._eval_chi2_jac
        }
      ).lowest_optimization_result

    else:

      # how to use basinhopping intelligently to keep things fast? single local minimize is ~100x faster...
      # maybe try single minimization at given seed first, and proceed to basin hopping if result is poor? e.g. didn't converge, or chi2 too big?
      self.opt_result = opt.minimize(
        self._eval_chi2,
        x0 = seeds,
        method = "BFGS",
        jac = self._eval_chi2_jac
      )
    
    # Extract the optimized parameters from the minimization result.
    self.p_opt = self._insert_fixed_params(self.opt_result.x)

    # Calculate the parameter covariance matrix, but only if the data had a covariance matrix -- otherwise not meaningful.
    if self.data.cov is not None:
      self.p_cov = np.linalg.inv(self._eval_chi2_hess(self.opt_result.x)) # TODO: should there be a 1/2 here or not????
      self.p_err = np.sqrt(np.diag(self.p_cov))
    else:
      self.p_cov = None
      self.p_err = [None] * len(self.p_opt)

    # Calculate the minimized chi2 and chi2/ndf.
    self.chi2 = self.opt_result.fun
    self.chi2_ndf = self.chi2 / self.ndf
    self.pval = self._eval_pval(self.opt_result.x)

    self.duration = time.perf_counter() - start_time

# ======================================================================================================================

  def _identify_linear_params(self):
    linear_param_indices = []
    for i in range(len(self.sp_params)):
      # Check if the 2nd derivative with respect to this parameter (H_ii) is identically zero.
      # Also check that the coeff. of this param. is independent of all other linear candidates so far, i.e. H_ij == 0.
      if self.sp_hess[i][i] == 0 and all([self.sp_hess[i][j] == 0 for j in linear_param_indices]):
        linear_param_indices.append(i)
    return linear_param_indices

# ======================================================================================================================

  # Expand the vector 'p' of floating parameter values into the full vector of all parameter values.
  def _insert_fixed_params(self, p_nonlinear):

    wrapped_p = self.fixed_template.copy()
    wrapped_p[self.where_float] = p_nonlinear

    jac = self.np_jac(self.data.x, *wrapped_p)[self.where_linear]
    M = jac @ (self.data.inv_cov @ jac.T)
    b = jac @ (self.data.inv_cov @ (self.np_expr_nonlinear(self.data.x, *p_nonlinear) - self.data.y))
    p_linear = np.linalg.inv(M) @ (-b)
    wrapped_p[self.where_linear] = p_linear

    return wrapped_p

# ======================================================================================================================

  # Compute the chi-squared at the given vector of parameter values.
  def _eval_chi2(self, p_nonlinear):
    p = self._insert_fixed_params(p_nonlinear)
    res = self.np_expr(self.data.x, *p) - self.data.y
    return res @ (self.data.inv_cov @ res)

# ======================================================================================================================

  # Compute the chi-squared jacobian vector (with respect to the parameters) at the given vector of parameter values.
  def _eval_chi2_jac(self, p_nonlinear):
    p = self._insert_fixed_params(p_nonlinear)
    res = self.np_expr(self.data.x, *p) - self.data.y
    jac = self.np_jac(self.data.x, *p)[self.where_float]
    return 2 * (jac @ (self.data.inv_cov @ res))

# ======================================================================================================================

  # Compute the chi-squared hessian matrix (with respect to the parameters) at the given vector of parameter values.
  def _eval_chi2_hess(self, p_nonlinear):
    p = self._insert_fixed_params(p_nonlinear)
    res = self.np_expr(self.data.x, *p) - self.data.y
    jac = self.np_jac(self.data.x, *p)#[self.where_float]
    hess = self.np_hess(self.data.x, *p)#[self.where_float][:, self.where_float]
    return 2 * (hess @ (self.data.inv_cov @ res) + jac @ (self.data.inv_cov @ jac.T))

# ======================================================================================================================

  # Calculate the two-sided p-value from the chi2 distribution with 'ndf' degrees of freedom.
  def _eval_pval(self, p_nonlinear):
    # The difference between this fit's chi2 and the mean of the distribution.
    mean_diff = abs(self._eval_chi2(p_nonlinear) - self.ndf)
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
      return jac.T @ (self.p_cov @ jac)

# ======================================================================================================================

  # Calculate the one-sigma error band of the optimal fitted curve evaluated at the given x values.
  def err(self, x):
    cov = self.cov(x)
    if cov is not None:
      return np.sqrt(np.diag(cov))

# ======================================================================================================================

  def fix(self, name, value):
    self.fixed[name] = value
    self._update_fixed()

  def free(self, name):
    del self.fixed[name]
    self._update_fixed()

  def _update_fixed(self):
    self.where_fixed.fill(False)
    self.where_float.fill(False)
    self.fixed_params.clear()
    self.float_params.clear()
    self.fixed_template.fill(np.nan)
    for i, param in enumerate(self.str_params):
      if param in self.fixed:
        self.where_fixed[i] = True
        self.where_float[i] = False
        self.fixed_params.append(param)
        self.fixed_template[i] = self.fixed[param]
      elif self.where_nonlinear[i]:
        self.where_float[i] = True
        self.float_params.append(param)
    self.ndf = len(self.data.y) - (len(self.sp_params) - len(self.fixed_params))
    self.err_chi2_ndf = np.sqrt(2 / self.ndf) # std. dev. of reduced chi2 distribution

# ======================================================================================================================

  def print(self, quality = True, parameters = True):

    lines = []

    if parameters:
      for i, param in enumerate(self.str_params):
        if param in self.fixed:
          lines.append(f"{io.format_value(param, self.fixed[param])} (fixed)")
        else:
          lines.append(io.format_value(param, self.p_opt[i], self.p_err[i]))

    if quality and self.data.cov is not None:
      lines += io.format_values(
        ("chi2/ndf", self.chi2_ndf, self.err_chi2_ndf),
        ("p-value", self.pval)
      )

    print(f"Fit completed in {self.duration:.{io.get_decimal_places(self.duration, 2)}f} seconds.")
    print(io.align(*lines, margin = 4))

# ======================================================================================================================

if __name__ == "__main__":

  std = 1

  x = np.linspace(0, 10, 200)
  y = (10 * np.cos(3*x) + x**2) + np.random.normal(0, std, size = len(x))
  err = np.ones(len(x)) * std
  data = Data(x, y, err = err)
  fit = Fit(data, "a * cos(b*x) + c * f(x)", definitions = {"f": x**2})
  fit.fit(hopping = True, guess = {"b": 3})
  fit.print()

  fit.fix("c", 1)
  fit.fix("a", 10)
  fit.fit(guess = {"b": 3.1})
  fit.print()

  fit.free("a")
  fit.fit(guess = {p: val for p, val in zip(fit.str_params, fit.p_opt)})
  fit.print()

  # plot = Plot()
  # plot.plot(x, y, err, line = None, label = "Data")
  # plot.plot(x, fit(x), fit.err(x), error_mode = "band", label = "Fit")
  # plot.databox(
  #   (r"$\chi^2$/ndf", fit.chi2_ndf, fit.err_chi2_ndf),
  #   ("$p$-value", fit.pval())
  # )
  # plot.labels(r"$\sigma$", "y", "Title")
  # plot.save("test.pdf")
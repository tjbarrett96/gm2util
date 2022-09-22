import sympy as sp
from sympy.utilities.lambdify import implemented_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
import scipy.sparse as sparse

from gm2util.data import Data
from gm2util.plot import Plot
import gm2util.io as io

import time
import re

# ======================================================================================================================

class Fit:

  def __init__(self, data, expr, arg = "x", definitions = None):

    # The Data object to fit, which wraps the numerical data as data.x, data.y, and data.cov.
    self.data = data
    
    # Get the inverse covariance matrix of the data.
    if self.data.cov is not None:
      if sparse.issparse(self.data.cov):
        self.inv_cov = sparse.diags(1 / self.data.cov.diagonal())
      else:
        self.inv_cov = np.linalg.inv(self.data.cov)
    else:
      self.inv_cov = np.eye(len(self.data.x))

    # Attach numerical definitions to custom functions named in the 'expr' string.
    # May be callable single-variable functions, or discrete numeric vectors which must match the shape of the data.
    sym_local_dict = {}
    if definitions is not None:
      for name, definition in definitions.items():
        if callable(definition):
          sym_local_dict[name] = implemented_function(name, definition)
        elif len(definition) == len(self.data.x):
          sym_local_dict[name] = implemented_function(name, lambda x: definition)
        else:
          raise ValueError()

    # Parse the model function and independent variable strings into SymPy expressions.
    self.sym_expr = sp.parse_expr(expr, local_dict = sym_local_dict)
    self.sym_arg = sp.Symbol(arg)

    # Identify the fit parameters as the set of all unbound symbols, minus the independent variable.
    # Convert the set into an ordered list, and sort by order of appearance within the user's function string.
    self.sym_params = sorted(
      list(self.sym_expr.free_symbols - {self.sym_arg}),
      key = lambda p: re.search(rf"\b{p.name}\b", expr).start()
    )
    self.str_params = [p.name for p in self.sym_params]

    # Compute the jacobian vector of first derivatives with respect to each parameter.
    self.sym_jac = [sp.diff(self.sym_expr, p) for p in self.sym_params]

    # Compute the hessian matrix of second derivatives with respect to each pair of parameters.
    self.sym_hess = [[sp.diff(self.sym_expr, p, q) for p in self.sym_params] for q in self.sym_params]

    # Identify which parameters in the sorted parameter list are linear.
    self.where_linear = np.full(len(self.sym_params), False)
    self.where_linear[self._identify_linear_params()] = True
    self.where_nonlinear = ~self.where_linear

    self.linear_params = [p for i, p in enumerate(self.sym_params) if self.where_linear[i]]
    self.nonlinear_params = [p for i, p in enumerate(self.sym_params) if self.where_nonlinear[i]]

    # Separate the symbolic expression into a linear part and nonlinear part (with respect to the parameters).
    self.sym_expr_linear = 0
    for i, param in enumerate(self.sym_params):
      if self.where_linear[i]:
        self.sym_expr_linear += self.sym_jac[i] * param
    self.sym_expr_nonlinear = self.sym_expr - self.sym_expr_linear

    self.fixed = {}
    self.where_fixed = np.full(len(self.sym_params), False)
    self.where_floating = np.full(len(self.sym_params), True)
    self.param_template = np.full(len(self.sym_params), np.nan)
    self.fixed_params = []
    self.floating_params = []
    self._update_fixed()

    # Convert SymPy expressions into NumPy functions.
    arg_list = [self.sym_arg, *self.sym_params]
    self.np_expr = Fit._sympy_to_numpy(arg_list, self.sym_expr)
    self.np_jac = Fit._sympy_to_numpy(arg_list, self.sym_jac)
    self.np_hess = Fit._sympy_to_numpy(arg_list, self.sym_hess)
    self.np_expr_nonlinear = Fit._sympy_to_numpy(
      [self.sym_arg] + [p for i, p in enumerate(self.sym_params) if self.where_nonlinear[i]],
      self.sym_expr_nonlinear
    )

# ======================================================================================================================

  # Helper function for converting a nested list of SymPy expressions into a NumPy function with the same structure.
  @staticmethod
  def _sympy_to_numpy(sym_args_list, sym_expr):
    if isinstance(sym_expr, list):
      return lambda x, *p: np.array([Fit._sympy_to_numpy(sym_args_list, item)(x, *p) for item in sym_expr])
    else:
      numpy_function = sp.lambdify(sym_args_list, sym_expr)
      # Ensure the function output matches the length of x. It won't by default if the function is independent of x.
      return lambda x, *p: numpy_function(x, *p) * np.ones(len(x))

# ======================================================================================================================

  # TODO: accept step sizes (setting scale for knowledge of initial seeds) for basin hopping
  def fit(self, guess = None, hopping = False):

    start_time = time.perf_counter()

    self.opt_result = None
    if len(self.floating_params) > 0:

      if guess is None:
        guess = {}
      if isinstance(guess, dict):
        seeds = np.ones(len(self.floating_params))
        for i, param in enumerate(self.floating_params):
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
    p_floating_opt = self.opt_result.x if self.opt_result is not None else []
    self.p_opt = self._expand_floating_params(p_floating_opt)

    # Calculate the parameter covariance matrix, but only if the data had a covariance matrix -- otherwise not meaningful.
    if self.data.cov is not None:
      self.p_cov = np.linalg.inv(self._eval_chi2_hess(p_floating_opt)) # TODO: should there be a 1/2 here or not????
      self.p_err = np.sqrt(np.diag(self.p_cov))
    else:
      self.p_cov = None
      self.p_err = [None] * len(self.p_opt)

    # Calculate the minimized chi2 and chi2/ndf.
    self.chi2 = self._eval_chi2(p_floating_opt)
    self.chi2_ndf = self.chi2 / self.ndf
    self.pval = self._eval_pval(p_floating_opt)

    self.duration = time.perf_counter() - start_time

# ======================================================================================================================

  def _identify_linear_params(self):
    linear_param_indices = []
    for i in range(len(self.sym_params)):
      # Check if the 2nd derivative with respect to this parameter (H_ii) is identically zero.
      # Also check that the coeff. of this param. is independent of all other linear candidates so far, i.e. H_ij == 0.
      if self.sym_hess[i][i] == 0 and all([self.sym_hess[i][j] == 0 for j in linear_param_indices]):
        linear_param_indices.append(i)
    return linear_param_indices

# ======================================================================================================================

  # Expand the vector of floating parameter values into the full vector of all parameters (incl. fixed and linear).
  def _expand_floating_params(self, p_floating):

    # Start with copy of parameter vector with fixed values filled-in, then fill in the supplied floating values.
    p_all = self.param_template.copy()
    p_all[self.where_floating] = p_floating

    # Invert the system of linear parameters, then fill them into the appropriate places in the parameter vector.
    jac = self.np_jac(self.data.x, *p_all)[self.where_linear]
    M = jac @ (self.inv_cov @ jac.T)
    b = jac @ (self.inv_cov @ (self.np_expr_nonlinear(self.data.x, *p_all[self.where_nonlinear]) - self.data.y))
    p_linear = np.linalg.inv(M) @ (-b)
    p_all[self.where_linear] = p_linear

    return p_all

# ======================================================================================================================

  # Compute the chi-squared at the given vector of parameter values.
  def _eval_chi2(self, p_floating):
    p_all = self._expand_floating_params(p_floating)
    res = self.np_expr(self.data.x, *p_all) - self.data.y
    return res @ (self.inv_cov @ res)

# ======================================================================================================================

  # Compute the chi-squared jacobian vector (with respect to the parameters) at the given vector of parameter values.
  def _eval_chi2_jac(self, p_floating):
    p_all = self._expand_floating_params(p_floating)
    res = self.np_expr(self.data.x, *p_all) - self.data.y
    jac = self.np_jac(self.data.x, *p_all)[self.where_floating]
    return 2 * (jac @ (self.inv_cov @ res))

# ======================================================================================================================

  # Compute the chi-squared hessian matrix (with respect to the parameters) at the given vector of parameter values.
  def _eval_chi2_hess(self, p_floating):
    p_all = self._expand_floating_params(p_floating)
    res = self.np_expr(self.data.x, *p_all) - self.data.y
    jac = self.np_jac(self.data.x, *p_all)
    hess = self.np_hess(self.data.x, *p_all)
    return 2 * (hess @ (self.inv_cov @ res) + jac @ (self.inv_cov @ jac.T))

# ======================================================================================================================

  # Calculate the two-sided p-value from the chi2 distribution with 'ndf' degrees of freedom.
  def _eval_pval(self, p_floating):
    # The difference between this fit's chi2 and the mean of the distribution.
    mean_diff = abs(self._eval_chi2(p_floating) - self.ndf)
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
    if (cov := self.cov(x)) is not None:
      return np.sqrt(cov.diagonal())

# ======================================================================================================================

  # Fix the given parameter name to the given value.
  def fix(self, name, value):
    self.fixed[name] = value
    self._update_fixed()

# ======================================================================================================================

  # Free the given parameter name from a previously fixed value.
  def free(self, name):
    del self.fixed[name]
    self._update_fixed()

# ======================================================================================================================

  # Update the internal state of fixed and floating parameters.
  def _update_fixed(self):

    # Reset lists identifying fixed and floating parameters.
    self.where_fixed.fill(False)
    self.where_floating.fill(False)
    self.fixed_params.clear()
    self.floating_params.clear()
    self.param_template.fill(np.nan)

    # For each parameter, check if fixed or floating and update internal state appropriately.
    for i, param in enumerate(self.str_params):
      if param in self.fixed:
        self.where_fixed[i] = True
        self.fixed_params.append(param)
        self.param_template[i] = self.fixed[param]
      elif self.where_nonlinear[i]:
        self.where_floating[i] = True
        self.floating_params.append(param)

    # Update the number of degrees of freedom, since the number of floating parameters may have changed.
    self._update_ndf()

# ======================================================================================================================

  # Update the number of degrees of freedom and one-sigma width of the reduced chi2 distribution.
  def _update_ndf(self):
    self.ndf = len(self.data.y) - (len(self.sym_params) - len(self.fixed_params))
    self.err_chi2_ndf = np.sqrt(2 / self.ndf)

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
  #   ("$p$-value", fit.pval)
  # )
  # plot.labels(r"$\sigma$", "y", "Title")
  # plot.save("test.pdf")
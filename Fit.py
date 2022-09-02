import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import re

# ======================================================================================================================

class Fit:

  def __init__(self, x, y, expr, arg = "x", cov = None):

    # The data to fit.
    self.x = x
    self.y = y
    self.cov = cov if cov is not None else np.ones(len(x))
    self.cov_inv = None

    # Select the method of computing the chi-squared, based on the shape of the data's uncertainties.
    if self.cov.ndim == 1:
      self.chi2 = self.chi2_err
      self.chi2_grad = self.chi2_grad_err
    elif self.cov.ndim == 2:
      self.cov_inv = np.linalg.inv(cov)
      self.chi2 = self.chi2_cov
      self.chi2_grad = self.chi2_grad_cov
    else:
      raise ValueError(f"Covariance matrix must have 1 or 2 dimensions, not {self.cov.ndim}.")

    # Parse the model function string into a SymPy expression.
    self.sp_expr = sp.parse_expr(expr)

    # Get the SymPy symbols for the independent variable and remaining fit parameters (sorted in order of appearance).
    self.sp_argument = sp.Symbol(arg)
    self.sp_parameters = sorted(
      list(self.sp_expr.free_symbols - {self.sp_argument}),
      key = lambda p: re.search(rf"\b{p.name}\b", expr).start()
    )

    # Convert the SymPy model function into a NumPy function.
    self.np_expr = sp.lambdify([self.sp_argument] + self.sp_parameters, self.sp_expr)

    # Compute the symbolic derivative of the model function with respect to each parameter, and convert to NumPy.
    self.sp_derivatives = [sp.diff(self.sp_expr, par) for par in self.sp_parameters]
    self.np_derivatives = [sp.lambdify([self.sp_argument] + self.sp_parameters, df_dp) for df_dp in self.sp_derivatives]

# ======================================================================================================================

  def fit(self, guess = None):

    self.guess = guess if guess is not None else np.ones(len(self.sp_parameters))

    # Minimize the chi-squared using BFGS, repeated with random steps in the initial conditions ("basin hopping").
    self.opt_result = opt.basinhopping(
      self.chi2,
      x0 = self.guess,
      minimizer_kwargs = {
        "method": "BFGS",
        "jac": self.chi2_grad
      }
    ).lowest_optimization_result
    
    self.p_opt = self.opt_result.x
    self.p_cov = self.opt_result.hess_inv

    self.min_chi2 = self.opt_result.fun
    self.ndf = len(self.y) - len(self.sp_parameters)
    self.chi2_ndf = self.min_chi2 / self.ndf

# ======================================================================================================================

  # Compute the chi-squared at the given vector of parameter values, using self.cov as a covariance matrix.
  def chi2_cov(self, p):
    res = self.np_expr(self.x, *p) - self.y
    return res.T @ self.cov_inv @ res

  # Compute the chi-squared at the given vector of parameter values, using self.cov as a vector of uncertainties.
  def chi2_err(self, p):
    res = self.np_expr(self.x, *p) - self.y
    return np.sum((res / self.cov)**2)

# ======================================================================================================================

  # Compute the chi-squared gradient vector (with respect to the parameters) at the given vector of parameter values.
  def chi2_grad_cov(self, p):
    res = self.np_expr(self.x, *p) - self.y
    return 2 * np.array([df_dp(self.x, *p).T @ self.cov_inv @ res for df_dp in self.np_derivatives])

  def chi2_grad_err(self, p):
    res = self.np_expr(self.x, *p) - self.y
    return 2 * np.array([np.sum(df_dp(self.x, *p) * res / self.cov**2) for df_dp in self.np_derivatives])
  
# ======================================================================================================================

  # Evaluate the model np function using the optimized parameters.
  def __call__(self, x):
    return self.np_expr(x, *self.p_opt)

# ======================================================================================================================

if __name__ == "__main__":

  std = 1

  # does a very bad job fitting cosine when initial guess is wrong! why?? must be a lot of very shallow local minima?

  x = np.linspace(0, 10, 1000)
  y = (10 * np.cos(3*x) + x**2) + np.random.normal(0, std, size = len(x))
  err = np.ones(len(x)) * std

  fit = Fit(x, y, "a * cos(b*x) + c * x**2", cov = err)
  fit.fit()
  print(fit.p_opt)
  print(np.sqrt(np.diag(fit.p_cov)))
  print(fit.chi2_ndf)

  plt.plot(x, y)
  plt.plot(x, fit(x))
  plt.show()
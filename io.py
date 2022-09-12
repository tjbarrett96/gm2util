import numpy as np

# ======================================================================================================================

def align(*lines, at = "=", margin = 0):
  split_lines = [line.split(at, 1) for line in lines]
  max_left_width = margin + max(len(split_line[0]) for split_line in split_lines)
  return "\n".join(
    f"{split_line[0].rjust(max_left_width)}{at}{split_line[1] if len(split_line) > 1 else ''}"
    for split_line in split_lines
  )

# ======================================================================================================================

def get_decimal_places(number, figures = 2):
  # Express the number as (d * 10^n), where 'd' is a single digit and 'n' is an integer.
  # Then log10(d * 10^n) == n + log10(d) --> floor(n + log10(d)) == n, since 0 < log10(d) < 1.
  if number != 0:
    first_fig_place = -int(np.floor(np.log10(abs(number))))
    return max(0, first_fig_place + (figures - 1))
  else:
    return 0

# ======================================================================================================================

def format_value(name, value, error = None, unit = None, math = False, decimals = None):

  # Determine how many decimal places are needed for the error to show 2 significant figures (or value, if no error).
  if decimals is None:
    decimals = get_decimal_places(error if error is not None else value, 2)

  pm = r"$\pm$" if math else "+/-"
  error_str = f" {pm} {error:.{decimals}f}" if error is not None else ""
  unit_str = f" {unit}" if unit is not None else ""

  # Generate the string "{name} = {value} +/- {error} {unit}".
  return f"{name} = {value:.{decimals}f}{error_str}{unit_str}"

def format_values(*lines, math = False):
  return [(format_value(*line, math = math) if isinstance(line, tuple) else line) for line in lines]

# ======================================================================================================================
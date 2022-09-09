import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ======================================================================================================================

class Plot:

  def __init__(
    self,
    width = 8,
    height = 5,
    left = 0.12,
    right = 0.93,
    bottom = 0.10,
    top = 0.93
  ):
    
    self.fig = plt.figure(figsize = (width, height))
    self.ax = plt.axes([left, bottom, right - left, top - bottom])

# ======================================================================================================================

  def plot(
    self,
    x,
    y,
    y_err = None,
    x_err = None,
    line = "-",
    marker = "o",
    error_mode = "bars",
    **kwargs
  ):
    if error_mode == "bars":
      return self.ax.errorbar(x, y, y_err, x_err, fmt = f"{marker if marker is not None else ''}{line if line is not None else ''}", ms = 4, capsize = 2, lw = 1, elinewidth = 0.5, **kwargs)
    elif error_mode == "band" and y_err is not None:
      band_plot = self.ax.fill_between(x, y - y_err, y + y_err, alpha = 0.25)
      line_plot = self.plot(x, y, None, x_err, line, None, "bars", **kwargs)
      return line_plot, band_plot
    else:
      raise ValueError(f"Plot error mode '{error_mode}' must be 'bars' or 'band'.")

# ======================================================================================================================

  def extend_x(self, factor = 0.15, left = False):
    left, right = self.ax.get_xlim()
    extend_amount = factor * (right - left)
    if left:
      left -= extend_amount
    else:
      right += extend_amount
    self.ax.set_xlim(left, right)

  def extend_y(self, factor = 0.1, bottom = False):
    bottom, top = self.ax.get_ylim()
    extend_amount = factor * (top - bottom)
    if bottom:
      bottom -= extend_amount
    else:
      top += extend_amount
    self.ax.set_ylim(bottom, top)

# ======================================================================================================================

  def legend(self, **kwargs):
    # Get the artist handles and text labels for everything in the current plot.
    handles, labels = self.ax.get_legend_handles_labels()
    # Make a dictionary mapping labels to handles; this ensures each label only appears with one handle.
    labels_to_handles = {label: handle for label, handle in zip(labels, handles)}
    # Make a legend, as long as there are some labels to show.
    if len(labels_to_handles) > 0:
      return self.ax.legend(handles = labels_to_handles.values(), labels = labels_to_handles.keys(), **kwargs)

# ======================================================================================================================

  def xlabel(self, label):
    return self.ax.set_xlabel(label, ha = "right", x = 1)

  def ylabel(self, label):
    return self.ax.set_ylabel(label, ha = "right", y = 1)

  def labels(self, xlabel, ylabel):
    self.xlabel(xlabel)
    self.ylabel(ylabel)
    self.legend()

# ======================================================================================================================

  @staticmethod
  def make_pdf(path):
    return PdfPages(path)

  def save(self, output):
    if isinstance(output, (str, PdfPages)):
      self.fig.savefig(output)
    else:
      raise ValueError(f"Figure output '{output}' invalid.")
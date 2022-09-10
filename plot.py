import matplotlib.pyplot as plt
import matplotlib as mpl
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
    
    # Create the figure and axes.
    self.fig = plt.figure(figsize = (width, height))
    self.ax = plt.axes([left, bottom, right - left, top - bottom])

    self.major_text_size = 14
    self.minor_text_size = 11

    # Configure major/minor axis ticks and background grid.
    self.ax.minorticks_on()
    self.ax.tick_params(which = "both", direction = "in", top = True, right = True, labelsize = self.minor_text_size)
    self.ax.tick_params(which = "major", length = 6)
    self.ax.tick_params(which = "minor", length = 3)
    self.ax.grid(alpha = 0.25)

    # Set the rule for switching tick labels to scientific notation.
    self.ax.ticklabel_format(scilimits = (-2, 5), useMathText = True)

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
      return self.ax.errorbar(
        x, y, y_err, x_err,
        fmt = f"{marker if marker is not None else ''}{line if line is not None else ''}",
        ms = 4, capsize = 2, lw = 1, elinewidth = 0.5, **kwargs
      )
    elif error_mode == "band" and y_err is not None:
      band_plot = self.ax.fill_between(x, y - y_err, y + y_err, alpha = 0.25)
      line_plot = self.plot(x, y, None, x_err, line, None, "bars", **kwargs)
      return line_plot, band_plot
    else:
      raise ValueError(f"Plot error mode '{error_mode}' must be 'bars' or 'band'.")

# ======================================================================================================================

  def draw_horizontal(self, y = 0, line = ":", color = "k", **kwargs):
    return self.ax.axhline(y, linestyle = line, color = color, **kwargs)

  def draw_vertical(self, x = 0, line = ":", color = "k", **kwargs):
    return self.ax.axvline(x, linestyle = line, color = color, **kwargs)

  def horizontal_spread(self, width, y = 0, color = "k", **kwargs):
    return self.ax.axhspan(y - width/2, y + width/2, color = color, alpha = 0.1, **kwargs)

  def vertical_spread(self, width, x = 0, color = "k", **kwargs):
    return self.ax.axvspan(x - width/2, x + width/2, color = color, alpha = 0.1, **kwargs)

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
      return self.ax.legend(
        handles = labels_to_handles.values(),
        labels = labels_to_handles.keys(),
        borderaxespad = 1,
        handlelength = 1,
        fontsize = self.minor_text_size,
        loc = "upper right",
        **kwargs
      )

# ======================================================================================================================

  def title(self, title):
    return self.ax.set_title(title, fontsize = self.major_text_size)

  def xlabel(self, label):
    return self.ax.set_xlabel(label, ha = "right", x = 1, fontsize = self.major_text_size)

  def ylabel(self, label):
    return self.ax.set_ylabel(label, ha = "right", y = 1, fontsize = self.major_text_size)

  def labels(self, xlabel, ylabel, title = None):
    if title is not None:
      self.title(title)
    self.xlabel(xlabel)
    self.ylabel(ylabel)
    self.legend()

# ======================================================================================================================

  # TODO: TeX rendering with tabular might work for alignment
  def databox(self, *lines, left = True):
    return self.ax.text(
      0.03 if left else 0.97,
      0.96,
      "\n".join(lines),
      ha = "left" if left else "right",
      va = "top",
      transform = self.ax.transAxes,
      fontsize = self.minor_text_size
    )

# ======================================================================================================================

  @staticmethod
  def make_pdf(path):
    return PdfPages(path)

  def save(self, output):
    if isinstance(output, (str, PdfPages)):
      self.fig.savefig(output)
    else:
      raise ValueError(f"Figure output '{output}' invalid.")
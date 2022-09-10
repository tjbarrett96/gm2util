
# ======================================================================================================================

def align(*lines, at = "=", margin = 0):
  split_lines = [line.split(at, 1) for line in lines]
  max_left_width = margin + max(len(split_line[0]) for split_line in split_lines)
  return "\n".join(
    f"{split_line[0].rjust(max_left_width)}{at}{split_line[1] if len(split_line) > 1 else ''}"
    for split_line in split_lines
  )

# ======================================================================================================================
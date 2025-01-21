import xylo
import xylo.types as t
import xylo.woods
import xylo.cut

import xylo.receptance as r
import xylo.sweep as xs

import xylo.loss.receptance as xlr

import xylo.tuning
import xylo.database as db

def sweep_find_length(wood, f, bar_template, freqd = 1, mind = 0.0001):
  l = 0.5
  ld = l / 2
  lf = None
  while ld > mind:
    bar = bar_template._replace(length = l)
    swp = xs.sweep(wood, bar, xylo.cut.none(bar), t.sweep_default)
    lf = swp.harmonics[0]
    if lf < f - freqd:
      l -= ld
    elif lf > f + freqd:
      l += ld
    else:
      return l, lf
    ld /= 2
  return l, lf

# overshoot: on test bar 200 F#5, the drilling and finishing reduced the frequency from 745 to 735. so, make the cut at 1.5% above the target frequency
def sweep_find_lengths(wood, notes, bar_template, tuning = xylo.tuning.yamaha_YX500R, overshoot = 1.015):
  sum_lengths = 0
  count = 0
  lengths = {}

  for n in notes:
    freq = tuning.yamaha_YX500R.note_to_freq(n)
    name = tuning.yamaha_YX500R.note_to_name(n)
    l, freq_est = sweep_find_length(wood, freq * overshoot, bar_template)
    print(f"*** note {name} ({n}): ***")
    print(f"    target freq {freq}, length {l}, est cut freq {freq_est}")
    sum_lengths += l
    count += 1
    lengths[n] = l

  print(f"total length {sum_lengths}, count {count}")
  return lengths

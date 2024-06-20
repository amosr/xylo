import math

import typing

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
notes88 = [(notes[(i - 40) % 12] + str((i+8) // 12)) for i in range(0,89)]

class Chromatic(typing.NamedTuple):
  reference: int
  reference_hz: float
  octave: int
  notes: dict[int,str]

  def note_to_freq(self, note):
    offset = note - self.reference
    return math.pow(2, offset / self.octave) * self.reference_hz

class Manual(typing.NamedTuple):
  tuning: Chromatic
  lengths: dict[int, float]

  def note_to_freq(self, note):
    return self.tuning.note_to_freq(note)
  def note_to_length(self, note):
    return self.lengths[note]
  def note_to_name(self, note):
    return notes88[note]

  def note_to_weights(self, note):
    fdl = self.note_to_freq(note)
    return ([1.0, 3, 6], [self.weight(fdl) * 1.0, self.weight(fdl * 3) * 0.3, self.weight(fdl * 6) * 0.1])
  def weight(self, freq):
    # 1 for frequencies below 10KHz
    # linear scale from 10KHz to 20KHz
    # 0 for frequencies above 20KHz
    return min(1.0, max(0.0, (20000 - freq) / 10000))

concert = Chromatic(49, 440, 12, notes88)
bright = concert._replace(reference_hz = 442)

# measurements of Yamaha YX-500R
yamaha_YX500R = Manual(bright, {
    45: 0.380,
    46: 0.375,
    47: 0.370,
    48: 0.365,
    49: 0.360,
    50: 0.355,
    51: 0.350,
    52: 0.340,
    53: 0.335,
    54: 0.330,
    55: 0.325,
    56: 0.320,
    57: 0.310,
    58: 0.305,
    59: 0.300,
    60: 0.295,
    61: 0.290,
    62: 0.285,
    63: 0.280,
    64: 0.270,
    65: 0.265,
    66: 0.260,
    67: 0.255,
    68: 0.250,
    69: 0.240,
    70: 0.235,
    71: 0.230,
    72: 0.225,
    73: 0.220,
    74: 0.215,
    75: 0.210,
    76: 0.200,
    77: 0.195,
    78: 0.190,
    79: 0.185,
    80: 0.180,
    81: 0.170,
    82: 0.165,
    83: 0.160,
    84: 0.155,
    85: 0.150,
    86: 0.145,
    87: 0.140,
    88: 0.130
})

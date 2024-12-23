import math

import typing

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
pos =   [0,   0.5,  1,   1.5,  2,   3,   3.5,  4,   4.5,  5,  5.5, 6 ]
notes88 = [(notes[(i - 40) % 12] + str((i+8) // 12)) for i in range(0,89)]
pos88 = [(pos[(i - 40) % 12] + ((i+8) // 12) * 7) - 5 for i in range(0,89)]


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
    # 1 for frequencies below 5KHz
    # linear scale from 5KHz to 15KHz
    # 0 for frequencies above 15KHz
    # based very loosely on http://supermediocre.org/wp-content/uploads/2015/10/Kori_tuning.png
    # which shows that it gets too difficult to tune even 3x at around C6~1051Hz
    return min(1.0, max(0.0, (15000 - freq) / 10000))

def note_to_x(note, relative_to = 1):
  return pos88[note] - pos88[relative_to]

def check_natural(note):
  return not('#' in notes88[note])

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

glock_F5_200_lengths = {57: 0.20898437500000006, 58: 0.20292968750000007, 59: 0.1970703125, 60: 0.19140625000000003, 61: 0.1859375, 62: 0.18066406250000006, 63: 0.1755859375, 64: 0.17050781250000002, 65: 0.16562500000000002, 66: 0.16093749999999998, 67: 0.15625, 68: 0.15175781250000003, 69: 0.1474609375, 70: 0.14316406250000008, 71: 0.13925781250000008, 72: 0.13515625, 73: 0.13125000000000003, 74: 0.12753906250000008, 75: 0.12382812500000002, 76: 0.12011718750000006, 77: 0.11699218750000004, 78: 0.11347656250000004, 79: 0.11035156250000003, 80: 0.10703125000000004, 81: 0.10410156250000002, 82: 0.10097656250000003, 83: 0.09804687500000002, 84: 0.09511718750000003, 85: 0.0923828125, 86: 0.08964843750000001, 87: 0.0873046875, 88: 0.0845703125}
glock_F5_200 = Manual(bright, glock_F5_200_lengths)

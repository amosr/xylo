# Some manual examples of tonality diamonds for now
import math

import typing
from xylo.database import Geometry

class Just(typing.NamedTuple):
  reference_hz: float
  rows: dict[str,float]

  def get_freqs(self):
    freq = {}
    for k,c in self.rows.items():
      freq[k] = c * self.reference_hz
    return freq

# octave reduction:
# for each ratio r, find by power of two p such that r*p in range [1,2)
#   (is power of two p necessarily unique? minimum if multiple?)
def octave_reduce(r):
  # require r > 0
  if r < 1:
    return octave_reduce(r * 2)
  elif r > 2:
    return octave_reduce(r / 2)
  else:
    return r

def mul(mul: float, l: dict[str,float]) -> list[float]:
  return { k: x * mul for k,x in l.items()}
# dict(map(lambda x: x * mul,l))

# 5-limit Meyer, Partch Genesis of a Music
diamond_5_meyer = Just(784, [
         [(4,3)],
     [(8,5), (5,3)],
  [(1,1), (5,5), (3,3)],
     [(5,4), (6,5)],
         [(3,2)]
])

# matrix with [1,5,3]. sort [1,3,5] wrt octave-reduction, ie sort by [ 1/1 -> 1, 3/2 -> 3, 5/4 -> 5 ]
matrix_153 = [
  [1/1, 1/5, 1/3],
  [5/1, 5/5, 5/3],
  [3/1, 3/5, 3/3]
]
# after octave reduction:
matrix_153_octave_red = [
  # 1/1 * 1 = 1/1
  # 1/5 * 8 = 8/5
  # 1/3 * 4 = 4/3
  [1/1, 8/5, 4/3],
  [5/4, 5/5, 5/3],
  [3/2, 6/5, 3/3]
]

# # 11-tonality diamond, partch, not laid out for marimba
# tdiamond_11_partch = Just(784, [
#   # lower 2/1
#   mul(0.5, [8/7]),
#   mul(0.5, [4/3, 9/7]),
#   mul(0.5, [16/11, 9/6, 10/7]),
#   mul(0.5, [8/5, 18/11, 5/3, 11/7]),
#   mul(0.5, [16/9, 9/5, 20/11, 11/6, 12/7]),
#   # higher 2/1
#   mul(1, [1/1, 9/9, 5/5, 11/11, 3/3, 7/7]),
#   mul(1, [9/8, 10/9, 11/10, 12/11, 7/6]),
#   mul(1, [5/4, 11/9, 6/5, 14/11]),
#   mul(1, [11/8, 12/9, 7/5]),
#   mul(1, [3/2, 14/9]),
#   mul(1, [7/4]),
# ])

# matrix for 11 tonality diamond
# row = [1,3,5,7,9,11]; sort wrt octave reduction
matrix_11 = [
  [1/1,  9/1,  5/1,  11/1,  3/1,  7/1],
  [1/9,  9/9,  5/9,  11/9,  3/9,  7/9],
  [1/5,  9/5,  5/5,  11/5,  3/5,  7/5],
  [1/11, 9/11, 5/11, 11/11, 3/11, 7/11],
  [1/3,  9/3,  5/3,  11/3,  3/3,  7/3],
  [1/7,  9/7,  5/7,  11/7,  3/7,  7/7],
]
# after octave reduction
matrix_11_octave_red = [
  [1/1,   9/8,   5/4,   11/8,  3/2,   7/4],
  [16/9,  9/9,   10/9,  11/9,  12/9,  14/9],
  [8/5,   9/5,   5/5,   11/10, 6/5,   7/5],
  [16/11, 18/11, 20/11, 11/11, 12/11, 14/11],
  [4/3,   9/6,   5/3,   11/6,  3/3,   7/6],
  [8/7,   9/7,   10/7,  11/7,  12/7,  7/7],
]

# marimba layout from genesis of a music 2e p261
diamond_11_partch = Just(784,
  # highest 2/1
  mul(   2, {'A1': 11/8}) | # 2156Hz
  mul(   2, {'B1': 9/8, 'B2': 11/10}) |
  # third 2/1
  mul(   1, {'C1': 7/4, 'C2': 9/5, 'C3': 11/6 }) |
  mul(   1, {'D1': 3/2, 'D2': 7/5,  'D3': 3/2,  'D4': 11/7}) |
  mul(   1, {'E1': 5/4, 'E2': 6/5,  'E3': 7/6,  'E4': 9/7,  'E5': 11/9}) |
  mul(   1, {'F1': 1/1, 'F2': 5/5,  'F3': 3/3,  'F4': 7/7,  'F5': 9/9, 'F6': 11/11}) | # neutral axis 784Hz
  # second 2/1
  mul(0.50, {'G1': 8/5, 'G2': 5/3,  'G3': 12/7, 'G4': 14/9, 'G5': 18/11}) |
  mul(0.50, {'H1': 4/3, 'H2': 10/7, 'H3': 4/3,  'H4': 14/11}) |
  mul(0.50, {'I1': 8/7, 'I2': 10/9, 'I3': 12/11}) |
  # lowest 2
  mul(0.25, {'J1': 16/9,'J2':  20/11}) |
  mul(0.25, {'K1': 16/11}) # 285Hz
)

diamond_11_partch_geom = {
  'A1': Geometry(length=119.0, width=32.0, depth=10.0, weight=1),
  'B1': Geometry(length=137.0, width=32.0, depth=10.0, weight=1),
  'B2': Geometry(length=134.0, width=32.0, depth=10.0, weight=1),
  'C1': Geometry(length=150.0, width=32.0, depth=10.0, weight=1),
  'C2': Geometry(length=152.0, width=32.0, depth=10.0, weight=1),
  'C3': Geometry(length=151.0, width=32.0, depth=10.0, weight=1),
  'D1': Geometry(length=166.0, width=32.0, depth=10.0, weight=1),
  'D2': Geometry(length=165.0, width=32.0, depth=10.0, weight=1),
  'D3': Geometry(length=167.0, width=32.0, depth=10.0, weight=1),
  'D4': Geometry(length=168.0, width=32.0, depth=10.0, weight=1),
  'E1': Geometry(length=186.0, width=32.0, depth=10.0, weight=1),
  'E2': Geometry(length=186.0, width=32.0, depth=10.0, weight=1),
  'E3': Geometry(length=186.0, width=32.0, depth=10.0, weight=1),
  'E4': Geometry(length=186.0, width=32.0, depth=10.0, weight=1),
  'E5': Geometry(length=185.0, width=32.0, depth=10.0, weight=1),
  'F1': Geometry(length=201.0, width=32.0, depth=10.0, weight=1),
  'F2': Geometry(length=200.0, width=32.0, depth=10.0, weight=1),
  'F3': Geometry(length=200.0, width=32.0, depth=10.0, weight=1),
  'F4': Geometry(length=201.0, width=32.0, depth=10.0, weight=1),
  'F5': Geometry(length=200.0, width=32.0, depth=10.0, weight=1),
  'F6': Geometry(length=200.0, width=32.0, depth=10.0, weight=1),
  'G1': Geometry(length=211.0, width=50.0, depth=10.0, weight=1),
  'G2': Geometry(length=211.0, width=50.0, depth=10.0, weight=1),
  'G3': Geometry(length=213.0, width=50.0, depth=10.0, weight=1),
  'G4': Geometry(length=212.0, width=50.0, depth=10.0, weight=1),
  'G5': Geometry(length=211.0, width=50.0, depth=10.0, weight=1),
  'H1': Geometry(length=232.0, width=50.0, depth=10.0, weight=1),
  'H2': Geometry(length=231.0, width=50.0, depth=10.0, weight=1),
  'H3': Geometry(length=232.0, width=50.0, depth=10.0, weight=1),
  'H4': Geometry(length=230.0, width=50.0, depth=10.0, weight=1),
  'I1': Geometry(length=261.0, width=50.0, depth=10.0, weight=1),
  'I2': Geometry(length=258.0, width=50.0, depth=10.0, weight=1),
  'I3': Geometry(length=260.0, width=50.0, depth=10.0, weight=1),
  'J1': Geometry(length=275.0, width=50.0, depth=10.0, weight=1),
  'J2': Geometry(length=275.0, width=50.0, depth=10.0, weight=1),
  'K1': Geometry(length=295.0, width=50.0, depth=10.0, weight=1)}


diamond_13_forster = Just(784, [
              [(13,8)],
            [(11,8), (13,10)],
          [(9,8), (11,10), (13,12)],
        [(7,4), (9,5), (11,6), (13,7)],
      [(3,2), (7,5), (3,2), (11,7), (13,9)],
    [(5,4), (6,5), (7,6), (9,7), (11,9), (13,11)],
  [(4,4), (5,5), (3,3), (7,7), (9,9), (11,11), (13,13)], # all 784Hz
    [(8,5), (5,3), (12,7), (14,9), (18,11), (22,13)],
      [(4,3), (10,7), (4,3), (14,11), (18,13)],
        [(8,7), (10,9), (12,11), (14,13)],
          [(16,9), (20,11), (24,13)],
# TODO this is listed as 2/1, but it's 392 = 784 * 1/2.
        [(1,2), (16,11), (20,13), (13,8)], # 392 Hz
          [(4,3), (16,13), (3,2)],
            [(1,4)] # 196 Hz
])

# bass marimba layout from genesis of a music
bass_partch = Just(784 / 4,
  mul(   1.00, {'A#3': 7/6})   | # 228.6Hz:  582x33 or
  mul(   1.00, {'A3':  9/8})   | # 220.5Hz:  564x33
  mul(   0.50, {'F#3': 11/6})  | # 179.6Hz:  561x33
  mul(   0.50, {'F3':  16/9})  | # 174.2Hz:  537x33
  mul(   0.50, {'D#3': 8/5})   | # 156.8Hz:  437x25, 566x33
  mul(   0.50, {'C#3': 16/11}) | # 142.54Hz: --X white mahogany 120x30x490
  mul(   0.50, {'A2':  8/7})   | # 112Hz:    --X blackbutt 136x33x640
  mul(   0.50, {'G2':  1/1})   | # 98Hz:     554x25, 720x33
  mul(   0.25, {'E2':  5/3})   | # 81.6Hz:   --/ oregon pine     515x35
  mul(   0.25, {'D2':  3/2})   | # 73.5Hz:   --X white mahogany? 830x33
  mul(   0.25, {'C2':  4/3})     # 65.3Hz:   --X reclaimed wharf? 680x25
)
# 228.6Hz:          582x33 or
# 220.5Hz:          564x33
# 179.6Hz:          561x33
# 174.2Hz:          537x33
# 156.8Hz:  437x25, 566x33
# 142.54Hz: --X white mahogany 120x30x490
# 112Hz:    --X blackbutt 136x33x640
# 98Hz:     554x25, 720x33, 621x30
# 81.6Hz:           789x33, 668x30
# 73.5Hz:   --X white mahog 702x30
# 65.3Hz:   --X reclaimed wharf? 680x25

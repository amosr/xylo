from typing import NamedTuple

import xylo.types as t

import numpy as np

import json


class Geometry(NamedTuple):
  length: float
  width: float
  depth: float
  weight: float

  def to_bar(self, elements = 300, min_depth_mul = 0.25) -> t.BarProps:
    l = np.mean(self.length) / 1000.0
    w = np.mean(self.width) / 1000.0
    d = np.mean(self.depth) / 1000.0
    return t.BarProps(width = w, length = l, depth = d, elements = elements, min_depth = d * min_depth_mul)

  def to_wood(self, base: t.Wood, coeffs) -> t.Wood:
    b = self.to_bar()
    wt = self.weight / 1000.0
    E = base.E * coeffs[0]
    G = base.G * coeffs[1]
    rho = (wt / (b.width * b.length * b.depth)) * coeffs[2]
    return t.Wood.make_E_G(E = E, G = G, rho = rho)

class Database:
  def __init__(self, geometries, notes):
    self.geometries = geometries
    self.notes = notes

  def get_geometry(self, note):
    return self.geometries[note]

  def get_bar(self, note, **kwargs):
    return self.geometries[note].to_bar(**kwargs)

  def get_wood(self, note, base: t.Wood = t.Wood.make_E_nu(E = 24.1e9, nu = 6.5, rho = 1000), coeffs = None):
    # json dumb, keys are strings
    if coeffs is None:
      coeffs = self.notes[str(note)]['wood']
    return self.geometries[note].to_wood(base, coeffs)

  def set_wood(self, note, coeffs):
    self.notes[str(note)]['wood'] = coeffs

  def get_best(self, note):
    n = self.notes[str(note)]
    vs = n['coeffs'].values()
    if len(vs) == 0:
      return None
    else:
      return min(vs, key = lambda c: c['loss'])

  def get_best_for_dims(self, note, num_dims):
    n = self.notes[str(note)]
    c = n['coeffs'].get(str(num_dims))
    if c is None:
      return None
    else:
      return c['coeff']

  def set_best_for_dims(self, note, num_dims, coeffs, loss):
    n = self.notes[str(note)]
    s = n['coeffs'].setdefault(str(num_dims), { 'coeff': coeffs, 'loss': loss })
    if s['loss'] > loss:
      s['coeff'] = coeffs
      s['loss'] = loss

  def clear_loss(self, note):
    n = self.notes[str(note)]
    c = n['coeffs']
    for d in c:
      c[d]['loss'] = 1e10

  def clear_loss_all(self):
    for n in self.notes.keys():
      self.clear_loss(n)

  def from_geometries(geometries):
    notes = {}
    for k in geometries:
      notes[str(k)] = { 'coeffs': {}, 'wood': [1.0, 1.0, 1.0] }
    return Database(geometries, notes)

  def from_obj(geometries, obj):
    return Database(geometries, obj['notes'])

  def from_file(geometries, fp = 'data/db.json'):
    with open(fp, 'r') as f:
      return Database.from_obj(geometries, json.load(f))

  def to_obj(self):
    return { 'notes': self.notes }

  def to_file(self, fp = 'data/db.json'):
    with open(fp, 'w') as f:
      json.dump(self.to_obj(), f, indent=2, sort_keys=True)

geometries = {
  # naturals
  57: Geometry([312.35, 312.95], [40.45, 40.40], [19.63, 19.66], 249),
  59: Geometry([302.62, 302.80], [40.45, 40.64], [19.56, 19.42], 254),
  61: Geometry([292.80, 292.87], [40.33, 40.41], [19.53, 19.58], 253),
  63: Geometry([281.87, 281.57], [40.36, 40.42], [19.49, 19.53], 225),
  64: Geometry([270.00, 269.80], [40.52, 40.50], [19.66, 19.49], 225),
  66: Geometry([261.20, 261.20], [40.37, 40.54], [19.55, 19.40], 219),
  68: Geometry([252.37, 252.16], [40.43, 40.27], [19.69, 19.49], 212),
  69: Geometry([240.56, 240.80], [40.30, 40.18], [19.40, 19.39], 202),
  71: Geometry([230.09, 230.54], [40.45, 40.44], [19.53, 19.53], 193),
  73: Geometry([220.94, 220.98], [40.26, 40.46], [19.48, 19.41], 177),
  # XXX: 75 needs to be shortened to <=206
  75: Geometry([208], [40.45, 40.7], [19.33, 19.58], 176),
  # 75: Geometry([206], [40.39, 40.61], [19.33, 19.58], 178),
  76: Geometry([201.76, 201.41], [40.25, 40.28], [19.41, 19.42], 163),
  78: Geometry([190.67, 190.64], [40.59, 40.46], [19.54, 19.46], 154),
  # XXX: 80 needs to be shortened to <=177
  80: Geometry([179], [40.55, 40.79], [19.54, 19.48], 152),
  # 80: Geometry([175], [40.55, 40.46], [19.33, 19.28], 155),
  81: Geometry([170.43, 170.65], [40.44, 40.36], [19.52, 19.44], 137),
  83: Geometry([162.45, 162.50], [40.36, 40.44], [19.53, 19.46], 129),
  85: Geometry([150.29, 150.33], [40.29, 40.25], [19.51, 19.39], 125),
  87: Geometry([140.27, 141.48], [40.42, 40.63], [19.45, 19.58], 117),
  88: Geometry([131.49, 132.07], [40.72, 40.45], [19.46, 19.52], 103),

  # accidentals
  58: Geometry([307.20, 307.18], [40.36, 40.73], [19.51, 19.45], 258),
  60: Geometry([296.02, 295.55], [40.59, 40.52], [19.48, 19.39], 241),
  62: Geometry([287.37, 286.64], [40.49, 40.41], [19.51, 19.47], 224),
  65: Geometry([266.45, 265.60], [40.24, 40.52], [19.49, 19.43], 221),
  67: Geometry([257.27, 257.66], [40.15, 40.17], [19.41, 19.52], 215),
  70: Geometry([235.92, 236.35], [40.44, 40.49], [19.38, 19.36], 197),
  72: Geometry([224.73, 225.04], [40.43, 40.27], [19.43, 19.51], 187),
  74: Geometry([216.38, 215.93], [40.49, 40.75], [19.52, 19.55], 177),
  77: Geometry([196], [40.22, 40.38], [19.22, 19.38], 166),
  # 77: Geometry([193.77, 194.13], [40.36, 40.34], [19.43, 19.35], 163),
  79: Geometry([185.99, 186.13], [40.35, 40.27], [19.42, 19.46], 149),
  82: Geometry([164.54, 164.81], [40.30, 40.53], [19.52, 19.56], 131),
  84: Geometry([156.42, 156.15], [40.60, 40.49], [19.49, 19.44], 133),
  86: Geometry([147.25, 146.69], [40.45, 40.34], [19.38, 19.33], 122),

  # extras
  # 1-layer tung weights
  45: Geometry([382.5], [36.36, 36.46], [19.61, 19.41], 300),
  46: Geometry([376], [36.19, 36.7], [19.49, 19.6], 287),
  47: Geometry([373], [35.9, 36.5], [19.5, 19.68], 281),
  48: Geometry([365], [36.83, 36.88], [19.47, 19.52], 277),
  49: Geometry([360], [37.07, 37.12], [19.39], 275),
  50: Geometry([357], [36.74, 37.08], [19.5, 19.4], 270),
  51: Geometry([350], [37.15, 36.75], [19.44, 19.49], 266),
  52: Geometry([340.5], [40.45, 40.68], [19.62, 19.4], 296),
  53: Geometry([334.9], [40.40], [19.53, 19.46], 278),
  54: Geometry([331.5], [40.30, 40.33], [19.56, 19.5], 270),
  55: Geometry([327], [40.37], [19.53, 19.56], 265),
  56: Geometry([319], [40.30, 40.65], [19.52, 19.45], 272),

}

def get(geometries = geometries, fp = 'data/db.json'):
  try:
    return Database.from_file(geometries, fp)
  except FileNotFoundError:
    return Database.from_geometries(geometries)

def with_db(act, geometries = geometries, fp = 'data/db.json'):
  db = get(geometries, fp)
  ret = act(db)
  db.to_file(fp)
  return ret
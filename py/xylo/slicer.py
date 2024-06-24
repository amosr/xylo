import typing
import jax.numpy as jnp
import xylo.types

import xylo.gcode.builder

import matplotlib.pyplot as plt

class Tool(typing.NamedTuple):
  radius: float # mm
  plunge: float # mm/m
  cut: float # mm/m
  move: float # mm/m
  spindle: float # rpm?

  passX: float # mm
  passY: float # mm
  passZ: float # mm

  liftZ: float # mm

class Slicer(typing.NamedTuple):
  tool: Tool

  def outline(self, bar: xylo.types.BarProps, sec: xylo.types.Sections):
    depth_shift = sec.depths # + self.tool.radius

    (nonzero,) = jnp.nonzero(sec.depths < bar.depth)
    start = nonzero[0]
    end = nonzero[-1]
    return (sec.xmids[start:end+1], depth_shift[start:end+1])

  def plot_outline(self, outline, dmax=32):
    (xs, ys) = outline
    ys = ys + self.tool.radius / 1000
    plt.scatter(xs, ys, s = 0.1, c = 'red')

    xss = []
    yss = []
    for d in range(0, dmax):
      dd = d / dmax
      xd = jnp.sin(dd * jnp.pi * 2) * self.tool.radius / 1000
      yd = jnp.cos(dd * jnp.pi * 2) * self.tool.radius / 1000
      xss.append(xs + xd)
      yss.append(ys + yd)
    plt.scatter(jnp.concat(xss), jnp.concat(yss), s = 0.1, c = 'green')

  # def toolpath_tangent(self, sec: xylo.types.Sections):
  #   gr = jnp.gradient(sec.depths)
  #   return (sec.xmids, sec.depths + self.tool.radius)

  # side-on cutting path
  def path(self, bar: xylo.types.BarProps, outline, precut = None):
    (xs, ys) = outline
    xs = (xs + bar.length / 2) * 1000
    ys = (bar.depth - ys) * 1000
    ysprecut = jnp.full_like(ys, 0) if precut is None else ((bar.depth - precut[1]) * 1000)
    x0 = xs[0]
    xS = xs[1] - x0
    xN = xs[-1]

    builder = xylo.gcode.builder.Builder()
    builder.c('G92', X = (x0 + xN) / 2, Y = bar.depth * 1000, Z = 0)
    builder.move(X = x0, Y = 0, Z = 0)

    zmin = -bar.width * 1000
    zstep = -self.tool.passZ

    for z in jnp.arange(0, zmin + zstep, zstep):
      z = max(z, zmin)
      builder.comment('plunge ' + str(z))
      builder.cut(X = x0, Y = ysprecut[0], Z = z, F = self.tool.plunge, S = self.tool.spindle)
      builder.comment('rough')
      for x in jnp.arange(x0, xN, self.tool.passX):
        ix = int((x - x0) / xS)
        y = ys[ix]
        y0 = ysprecut[ix]
        builder.cut(X = x, Y = y, Z = z, F = self.tool.cut, S = self.tool.spindle)
        builder.cut(X = x, Y = y0, Z = z, F = self.tool.cut, S = self.tool.spindle)
      builder.comment('cleanup')
      for ix in jnp.arange(len(xs) - 1, 0, -1):
        x = xs[ix]
        y = ys[ix]
        builder.cut(X = x, Y = y, Z = z, F = self.tool.cut, S = self.tool.spindle)
      builder.move(X = x0, Y = ysprecut[0], Z = z)

    return builder

  # top-down cutting path
  def topdown_depths(self, bar: xylo.types.BarProps, outline, precut = None):
    (xs, ys) = outline
    xs = (xs + bar.length / 2) * 1000
    ys = (bar.depth - ys) * 1000
    ysprecut = jnp.full_like(ys, 0) if precut is None else ((bar.depth - precut[1]) * 1000)
    x0 = xs[0]
    xS = xs[1] - x0
    xN = xs[-1]

    dxs = []
    dds = []
    d0s = []
    for x in jnp.arange(x0, xN+self.tool.passX, self.tool.passX):
      ix_pre = max(0, int((x - x0 - self.tool.radius) / xS))
      ix_pst = int((x - x0 + self.tool.radius) / xS) + 1
      d = jnp.min(ys[ix_pre:ix_pst])
      d0 = jnp.min(ysprecut[ix_pre:ix_pst])
      dxs.append(x)
      dds.append(d)
      d0s.append(d0)
    return (dxs, dds, d0s)

  # top-down cutting path
  def path_topdown(self, bar: xylo.types.BarProps, outline, precut = None, cut_width = None):
    (dxs, dds, d0s) = self.topdown_depths(bar, outline, precut)
    x0 = dxs[0]
    xN = dxs[-1]

    bar_width = bar.width * 1000
    if cut_width is None:
      cut_width = bar_width
    yrng = jnp.arange(-(bar_width / 2 - cut_width / 2), -(bar_width / 2 + cut_width / 2), -self.tool.passY)

    builder = xylo.gcode.builder.Builder()
    builder.c('G92', X = self.tool.radius, Y = -self.tool.radius, Z = 0)
    builder.move(Z = self.tool.liftZ, F = self.tool.move)
    builder.move(X = x0)
    builder.move(Y = -(bar_width / 2 - cut_width / 2))

    pred = 0
    prex = x0 - self.tool.passX
    for i in range(0, len(dxs)):
      x  = dxs[i]
      d  = -dds[i]
      d0 = -d0s[i]
      builder.comment(f'x {x}')
      yrng_iter = yrng
      if i % 2 == 1: yrng_iter = jnp.flip(yrng)

      for y in yrng_iter:
        builder.move(X = x, Y = y, Z = d0, F = self.tool.move)
        builder.cut(X = x, Y = y, Z = d, F = self.tool.plunge)
        builder.cut(X = prex, Y = y, Z = pred, F = self.tool.cut)
        builder.move(X = prex, Y = y, Z = d0, F = self.tool.move)

      prex = x
      pred = d

    builder.move(Z = self.tool.liftZ)
    builder.c('M05')
    builder.move(Y = -self.tool.radius)
    builder.move(X = self.tool.radius)
    return builder


tool8 = Tool(radius = 25.4 / 8 / 2, plunge = 500, cut = 500, move = 500, spindle = 10000, passX = 2.0, passY=1.5, passZ = 1.5, liftZ = 1.0)
# tool8 = Tool(radius = 25.4 / 8 / 1000 / 2, plunge = 50, cut = 100, spindle = 10000, passX = 1.5, passZ = 1.5)
slicer8 = Slicer(tool8)

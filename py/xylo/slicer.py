import typing
import jax.numpy as jnp
import xylo.types

import xylo.gcode.builder

import matplotlib.pyplot as plt

class Tool(typing.NamedTuple):
  radius: float # mm
  plunge: float # mm/s
  cut: float # mm/s
  spindle: float # rpm?

  passX: float # mm
  passZ: float # mm

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
    ys = ys + self.tool.radius
    plt.scatter(xs, ys, s = 0.1, c = 'red')

    xss = []
    yss = []
    for d in range(0, dmax):
      dd = d / dmax
      xd = jnp.sin(dd * jnp.pi * 2) * self.tool.radius
      yd = jnp.cos(dd * jnp.pi * 2) * self.tool.radius
      xss.append(xs + xd)
      yss.append(ys + yd)
    plt.scatter(jnp.concat(xss), jnp.concat(yss), s = 0.1, c = 'green')

  # def toolpath_tangent(self, sec: xylo.types.Sections):
  #   gr = jnp.gradient(sec.depths)
  #   return (sec.xmids, sec.depths + self.tool.radius)

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

tool8 = Tool(radius = 25.4 / 8 / 1000 / 2, plunge = 12, cut = 26, spindle = 10000, passX = 1.5, passZ = 1.5)
slicer8 = Slicer(tool8)

# Loss based on weighted receptance
# Differentiable
import jax
import jax.numpy as jnp

import xylo.types as t
import xylo.receptance as r
import xylo.cut

def loss(cut: t.CutCubic, wood: t.Wood, bar: t.BarProps, fundamental: float, ws: jnp.ndarray):
    sections = xylo.cut.cubic(bar, cut)
    wsT = ws.T
    rs  = jax.vmap(lambda w: r.receptance_scalar(wood, bar, sections, w))(wsT[0] * 2 * jnp.pi * fundamental)
    # jax.debug.print("{rs}", rs=rs)
    return -jnp.dot(rs, wsT[1])

def loss_(cut: t.CutCubic, args):
  wood, bar, fundamental, ws = args
  return loss(cut, wood, bar, fundamental, ws)

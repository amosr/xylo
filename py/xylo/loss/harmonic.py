# Loss based on computing whole harmonics
# Not differentiable
import jax
import jax.numpy as jnp

import xylo.types as t
import xylo.sweep
import xylo.cut

from functools import partial


@partial(jax.jit, static_argnames=['consts', 'shape', 'sweep_opts'])
def loss(cut: t.CutCubic, wood: t.Wood, bar: t.BarProps, sweep_opts: t.FrequencySweep, target: jnp.ndarray, weights: jnp.ndarray = jnp.array([1.0, 0.5, 0.25])):
    sections = xylo.cut.cubic(bar, cut)
    # jax.debug.print("{c}; {b}", c = cut, b = bar.depths)
    sweep = xylo.sweep.sweep(wood, bar, sections, sweep_opts)
    diff = jnp.abs((target - sweep.harmonics) / target)
    loss = diff * weights
    # jax.debug.print("{c} T {t}; A {h}: loss {d}", c = cut, t = target, h = sweep.harmonics, d = loss)
    return jnp.sum(loss)

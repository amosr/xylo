# Loss based on computing whole harmonics
# Not differentiable
import jax
import jax.numpy as jnp

import xylo.types as t
import xylo.sweep
import xylo.cut

from functools import partial


@partial(jax.jit, static_argnames=['bar', 'sweep_opts'])
def loss(sections: t.CutCubic, wood: t.Wood, bar: t.BarProps, sweep_opts: t.FrequencySweep, target: jnp.ndarray, weights: jnp.ndarray = jnp.array([1.0, 0.3, 0.15])):
    sweep = xylo.sweep.sweep(wood, bar, sections, sweep_opts)
    diff = jnp.abs((target - sweep.harmonics) / target)
    loss = diff * weights
    return jnp.sum(loss)

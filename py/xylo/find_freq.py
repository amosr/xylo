# Find natural frequencies of cut bars using receptance
import jax
import jax.numpy as jnp
import numpy as np

import xylo.types as t
import xylo.receptance as r

# def sweep():

def bracket(sweep: jnp.ndarray, recepts: jnp.ndarray, size = 3):
  """Find the peaks of higher receptance in a sweep"""
  sweeprecepts = jnp.stack([sweep, recepts]).T

  def bracket_go(pre, now):
    # (pre, k) = prek
    (pre_freq, pre_v) = (pre[0], pre[1])
    (now_freq, now_v) = (now[0], now[1])
    return (now, jnp.array([pre_freq, now_freq, jnp.logical_and(pre_v >= 0, now_v < 0)]))
  _, brackets = jax.lax.scan(bracket_go, jnp.array([-1.0, -1.0]), sweeprecepts)
    
  nz = brackets[brackets[:,2].nonzero(size = size, fill_value = 0)]
  return nz

def find_freq(wood: t.Wood, bar: t.BarProps, sections: t.Sections, bracket: jnp.ndarray, iters = 40):
  wlo = bracket[0]
  whi = bracket[1]
  ylo = r.receptance_scalar(wood, bar, sections, 2 * jnp.pi * wlo)
  yhi = r.receptance_scalar(wood, bar, sections, 2 * jnp.pi * whi)

  estimate = whi
  dw = wlo - whi
  state0 = jnp.array([estimate, dw])
  def go(k: int, state: jnp.ndarray) -> jnp.ndarray:
      estimate = state[0]
      dw = state[1] * 0.5
      wmid = estimate + dw
      ymid = r.receptance_scalar(wood, bar, sections, 2 * jnp.pi * wmid)
      estimate = jnp.where(ymid <= 0, wmid, estimate)
      # jax.debug.print("{estimate}: {ymid}", estimate=estimate, ymid=ymid)
      return jnp.array([estimate, dw])
      
  return jax.lax.fori_loop(0, iters, go, state0).T[0]

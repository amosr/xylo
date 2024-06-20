import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import evosax

from typing import NamedTuple, Optional
from functools import partial
import math
import json

import xylo
import xylo.types as t
import xylo.woods
import xylo.cut

import xylo.receptance as r
import xylo.sweep as xs

import xylo.loss.receptance as xlr
import xylo.loss.harmonic as xlh

import xylo.tuning


class Options(NamedTuple):
  num_generations: int = 100
  rng = jax.random.key(0)
  strategy: any = evosax.OpenES(popsize = 2000, num_dims = 3)
  params = strategy.default_params.replace(init_min = 0, init_max = 1, clip_min = 0, clip_max = 1)
  init_mean: Optional[jnp.ndarray] = None
  absolute_tolerance: float = 1e-5


def optimize_geometry(bar: t.BarProps, wood: t.Wood, options: Options, fundamental: float, partials: jnp.ndarray = jnp.array([1.0, 3.0, 6.0]), weights: jnp.ndarray = jnp.array([1.0, 0.3, 0.15])):
  sweep_opt = t.FrequencySweep(start_freq = fundamental * 0.1, stop_freq = fundamental * 10, num_freq = 30, bisect_iters = 10)
  def cut(s):
    return xylo.cut.spline(bar, s)
  def loss(s):
    return xlh.loss(cut(s), wood, bar, sweep_opt, fundamental * partials, weights)

  rng = options.rng
  state = options.strategy.initialize(rng, options.params, init_mean = options.init_mean)
  for i in range(options.num_generations):
    rng, rng_gen = jax.random.split(rng, 2)
    x, state = options.strategy.ask(rng_gen, state, options.params)
    fitness = jax.vmap(loss, in_axes = 0)(x)
    fitness = jnp.float32(fitness)
    state = options.strategy.tell(x, fitness, state, options.params)

    if abs(state.best_fitness) < options.absolute_tolerance:
      print(f"iteration {i}: reached fitness {state.best_fitness}")
      return state

    if i % 10 == 0:
      print(f"iteration {i}")
      print(state.best_member, state.best_fitness)
      sections = cut(state.best_member)
      # print(sections.depths)
      swp = xs.sweep(wood, bar, sections, t.sweep_default)
      print(swp.harmonics, swp.harmonics / fundamental, bar.length)

  return state


def optimize_wood(bar: t.BarProps, wood_base: t.Wood, options: Options, partials_measured: jnp.ndarray):
  sweep_opt = t.sweep_default
  sections = xylo.cut.none(bar)

  def wood(w):
    return t.Wood.make_E_G(E = wood_base.E * w[0], G = wood_base.G * w[1], rho = wood_base.rho * w[2])

  def loss(w):
    return xlh.loss(sections, wood(w), bar, sweep_opt, partials_measured)

  rng = options.rng
  state = options.strategy.initialize(rng, options.params, init_mean = options.init_mean)
  for i in range(options.num_generations):
    rng, rng_gen = jax.random.split(rng, 2)
    x, state = options.strategy.ask(rng_gen, state, options.params)
    fitness = jax.vmap(loss, in_axes = 0)(x)
    fitness = jnp.float32(fitness)
    state = options.strategy.tell(x, fitness, state, options.params)

    if abs(state.best_fitness) < options.absolute_tolerance:
      print(f"iteration {i}: reached fitness {state.best_fitness}")
      return state

    if i % 10 == 0:
      print(f"iteration {i}")
      print(wood(state.best_member), state.best_fitness)
      # print(sections.depths)
      swp = xs.sweep(wood(state.best_member), bar, sections, t.sweep_default)
      print(swp.harmonics, (swp.harmonics - partials_measured) / partials_measured, bar.length)

  return state, wood(state.best_member)

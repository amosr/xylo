# Search for harmonics of cut bars using receptance
import jax
import jax.numpy as jnp
import numpy as np

from typing import NamedTuple

import xylo.types as t
import xylo.receptance as r
import xylo.find_freq as ff


class SweepResult(NamedTuple):
    sweep: jnp.ndarray
    recepts: jnp.ndarray
    harmonics: jnp.ndarray

def sweep(wood: t.Wood, bar: t.BarProps, sections: t.Sections, sweep_opts: t.FrequencySweep):
    sweep = jnp.logspace(jnp.log10(sweep_opts.start_freq), jnp.log10(sweep_opts.stop_freq), sweep_opts.num_freq)
    recepts = jax.vmap(lambda w: r.receptance_scalar(wood, bar, sections, 2 * jnp.pi * w))(sweep)
    # plt.loglog(sweep / (2 * jnp.pi), recepts)
    # plt.semilogx(sweep / (2 * jnp.pi), recepts)
    # plt.plot(sweep, recepts)
    # print(jnp.min(recepts), jnp.max(recepts))

    brackets = ff.bracket(sweep, recepts)
    freqs = jax.vmap(lambda br: ff.find_freq(wood, bar, sections, br))(brackets)
    return SweepResult(sweep = sweep, recepts = recepts, harmonics = freqs)
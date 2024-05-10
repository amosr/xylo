import jax.numpy as jnp

from xylo.types import CutCubic, BarProps, Sections


def cubic(bar: BarProps, cut: CutCubic) -> Sections:
    # length of each section
    l = bar.length / bar.elements
    # midpoint of each section
    x = jnp.linspace(0 + l/2, bar.length / 2 - l/2, bar.elements // 2)
    # cut-out
    y = cut.cubic * (x ** 3) + cut.square * (x ** 2) + cut.offset
    # clamped
    yclamp = jnp.maximum(jnp.minimum(y, bar.depth), bar.min_depth)
    
    xmids = jnp.append( - jnp.flip(x), x)
    depths = jnp.append( jnp.flip(yclamp), yclamp)
    return Sections(xmids = xmids, depths = depths, length_per_element = l)
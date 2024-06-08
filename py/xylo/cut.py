import jax.numpy as jnp
import jax_cosmo.scipy.interpolate as interpolate

from xylo.types import CutCubic, BarProps, Sections


# cubic without linear component; zhao 2011 does this to ensure second derivative is zero at x=0
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

# cubic with linear component, semi-normalised inputs
def cubic_linear(bar: BarProps, cut: jnp.ndarray) -> Sections:
    # length of each section
    l = 2 / bar.elements
    # midpoint of each section
    x = jnp.linspace(0 + l/2, 1 - l/2, bar.elements // 2)
    # cut-out
    y = cut[3] * (x ** 3) + cut[2] * (x ** 2) + cut[1] * x + cut[0]
    # clamped
    ydepth = bar.depth - y * (bar.depth - bar.min_depth)
    yclamp = jnp.maximum(jnp.minimum(ydepth, bar.depth), bar.min_depth)
    xmids = jnp.append( - jnp.flip(x), x)
    depths = jnp.append( jnp.flip(yclamp), yclamp)
    return Sections(xmids = xmids, depths = depths, length_per_element = l)

def quartic(bar: BarProps, cut: jnp.ndarray) -> Sections:
    # length of each section
    l = bar.length / bar.elements
    # midpoint of each section
    x = jnp.linspace(0 + l/2, bar.length / 2 - l/2, bar.elements // 2)
    # cut-out
    y = cut[3] * (x ** 4) + cut[2] * (x ** 3) + cut[1] * (x ** 2) + cut[0]
    # clamped
    yclamp = jnp.maximum(jnp.minimum(y, bar.depth), bar.min_depth)
    
    xmids = jnp.append( - jnp.flip(x), x)
    depths = jnp.append( jnp.flip(yclamp), yclamp)
    return Sections(xmids = xmids, depths = depths, length_per_element = l)

def spline(bar: BarProps, cut: jnp.ndarray, max_spread: float = 0.8, symmetric: bool = True) -> Sections:
    cut_depths, spread = cut[0:-1], cut[-1]
    if symmetric:
        cut_depths = jnp.append(cut_depths, jnp.flip(cut_depths))
    cut_depths = jnp.concatenate([jnp.array([1.0]), cut_depths, jnp.array([1.0])])
    # length of each section
    l = bar.length / bar.elements
    bl2 = bar.length/2
    cutx = bl2 * spread * max_spread
    x = jnp.linspace(-bl2 + l/2, bl2 - l/2, bar.elements)

    xi = jnp.linspace(-cutx, cutx, len(cut_depths))

    yi = bar.min_depth + cut_depths * (bar.depth - bar.min_depth)

    i = interpolate.InterpolatedUnivariateSpline(xi, yi, k=2)
    y = jnp.where(jnp.logical_and(x <= cutx, x >= -cutx), i(x), bar.depth)

    yclamp = jnp.maximum(jnp.minimum(y, bar.depth), bar.min_depth)
    
    return Sections(x, yclamp, length_per_element = l)

def spline_scale(cut: jnp.ndarray, mul: float) -> jnp.ndarray:
    cut_depths, spread = cut[0:-1], cut[-1]
    return jnp.concatenate([1 - (1 - cut_depths) * mul, jnp.array([spread])])
     # jnp.concatenate([1 - (1 - spline[0:-1]) * 0.2, jnp.array([spline[-1]])]))

def none(bar: BarProps) -> Sections:
    return spline(bar, jnp.array([1.0, 1.0, 1.0]))
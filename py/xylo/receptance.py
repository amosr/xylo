# Timoshenko beam receptance
import jax
import jax.numpy as jnp

import xylo.types as t


def receptance(wood: t.Wood, bar: t.BarProps, sections: t.Sections, w: float, shear_factor: float = 5/6):
    I = bar.width * sections.depths ** 3 / 12
    EI = wood.E * I
    area = bar.width * sections.depths
    
    lens = jnp.full_like(sections.depths, sections.length_per_element)
    
    aplus = 1 + wood.E / (shear_factor * wood.G)
    aminus = 1 - wood.E / (shear_factor * wood.G)
    
    flipsign = w * w < shear_factor * wood.G * area / (wood.rho * I)
    sign = jnp.where(flipsign, 1, -1)
    
    m   = wood.rho * I * w * w
    n   = jnp.sqrt(m * (m * aminus ** 2 + 4 * wood.E * area))
    eta = jnp.sqrt(((-sign * m * aplus) + sign * n) / 2 * EI)
    xi  = jnp.sqrt(((m * aplus) + n) / (2*EI))
    s   = wood.rho * w * w / (shear_factor * wood.G * xi) - xi
    r   = wood.rho * w * w / (shear_factor * wood.G * eta) + sign * eta
    e   = r * eta ** 2 / (s * xi ** 2)
    xi_len  = xi * lens
    eta_len = eta * lens
    c   = jnp.cos(xi_len)
    ch  = jnp.cosh(eta_len)
    si  = jnp.sin(xi_len)
    sh  = jnp.sinh(eta_len)
    f1  = si * sh
    f2  = c * ch
    f3  = f2 - 1
    f4  = si * ch
    f5  = c * sh
    f6  = c - ch
    recip_delta = 1 / (wood.rho * w * w * area * (2 - 2 * f2 + sign * (e + -sign / e)*f1))
    
    b11 = -sign * f5 * (r * eta ** 2 / (s * xi) - eta) + f4 * (s * xi ** 2 / (r * eta) - xi)
    b12 = -sign * f3 * (s * xi + r * eta) + -sign * f1 * (r * eta ** 2 / xi + -sign * s * xi ** 2 / eta)
    b22 = f4 * r * eta * (s - r * eta / xi) + f5 * s * xi * (r - s * xi / eta)

    # add all of the remaining sections in section C
    # signs_ref=[-1 1 1 1 -1 -1 -1 -1 -1 1   1 -1 -1  1];
    #             1 2 3 4  5  6  7  8  9 10 11 12 13 14
    gamma1_11 = b11 # -sign * f5 * (r * eta ** 2 / (s * xi) - eta) + f4 * (s * xi ** 2 / (r * eta) - xi)
    gamma1_12 = -b12 # sign * f3 * (s * xi + r * eta) + sign * f1 * (r * eta ** 2 / xi + -sign * s * xi ** 2 / eta)
    gamma1_22 = b22 # f4 * r * eta * (s - r * eta / xi) + f5 * s * xi * (r - s * xi / eta)
    
    gamma1 = recip_delta * jnp.array([[ gamma1_11, gamma1_12 ], [ gamma1_12, gamma1_22 ]])
    
    gamma2_11 = -sign * sh * (r * eta ** 2 / (s * xi) - eta) + si * (s * xi ** 2 / (r * eta) - xi)
    gamma2_12 = -sign * f6 * (r * eta - s * xi)
    gamma2_21 = -gamma2_12 # sign * f6 * (r * eta - s * xi)
    gamma2_22 = sh * (r * s * xi - s ** 2 * xi ** 2 / eta) + si * (r * s * eta - r ** 2 * eta ** 2 / xi)
    gamma2 = recip_delta * jnp.array([[ gamma2_11, gamma2_12 ], [ gamma2_21, gamma2_22 ]])
    
    gamma3 = gamma1 * jnp.array([[ [1], [-1] ], [ [-1], [1]]])
    
    beta0 = recip_delta[0] * jnp.array([ [ b11[0], b12[0] ], [ b12[0], b22[0] ] ])
    
    result = jax.lax.fori_loop(1, bar.elements, lambda k, beta: gamma3.T[k] - gamma2.T[k].T * jnp.linalg.solve(gamma1.T[k] + beta, gamma2.T[k]), beta0)
    return result

def receptance_scalar(wood: t.Wood, bar: t.BarProps, sections: t.Sections, w: float):
    return receptance(wood, bar, sections, w)[0,0]
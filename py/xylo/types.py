# import jax
import jax.numpy as jnp
# import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

# from functools import partial

# jax.config.update("jax_enable_x64", True)

class Wood(NamedTuple):
    """Physical properties of the wood"""

    def make_E_G(rho: float, E: float, G: float):
        return Wood(rho = rho, E = E, G = G, nu = (E / (2 * G)) - 1)
    def make_G_nu(rho: float, G: float, nu: float):
        return Wood(rho = rho, E = 2 * G * (1 + nu), G = G, nu = nu)
    def make_E_nu(rho: float, E: float, nu: float):
        return Wood(rho = rho, E = E, G = E / (2 * (1 + nu)), nu = nu)
    
    rho: float
    """Density (kg/m^3)"""
    E: float
    """Young's modulus (Pa) or elastic modulus:
    https://en.wikipedia.org/wiki/Young%27s_modulus
    Typically, wood is on the order of 10 gigapascals (10e9)
    E = 2 * G * (1 + nu)"""
    G: float
    """Shear modulus (Pa):
    https://en.wikipedia.org/wiki/Shear_modulus
    Typically, wood is on the order of 4 gigapascals (4e9)"""
    nu: float
    """Poisson's ratio:
    https://en.wikipedia.org/wiki/Poisson%27s_ratio
    Most materials between 0.0 and 0.5; typically 0.2 to 0.3"""


class CutCubic(NamedTuple):
    """Cut coefficients: these are the variables we optimise"""
    cubic:   float
    square:  float
    offset:  float

class BarProps(NamedTuple):
    """Properties about the bar: these are constant for each note"""
    width: float
    depth: float
    length: float
    elements: int
    min_depth: float
    # assert elements % 2 = 0

    # def num_subdivisions(self):
    #     return (self.elements // 2) + 1

class Sections(NamedTuple):
    """The result of cutting a bar"""
    xmids: jnp.ndarray
    depths: jnp.ndarray
    length_per_element: float

    def plot(self, to_scale=True):
        plt.plot(self.xmids, self.depths)
        plt.plot(self.xmids, jnp.full(self.xmids.shape, 0))
        if to_scale:
            plt.gca().set_aspect('equal')

class FrequencySweep(NamedTuple):
    start_freq: float
    stop_freq: float
    num_freq: int
    bisect_iters: int
    num_harmonics: int = 3

    def sweep(self):
        return jnp.logspace(jnp.log10(self.start_freq), jnp.log10(self.stop_freq), self.num_freq)


# C8 is 4KHz and we tune to sixth partial, so run sweep to around 30KHz
sweep_default = FrequencySweep(start_freq = 100, stop_freq = 30000, num_freq = 100, bisect_iters = 16)
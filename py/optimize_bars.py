import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import evosax

import json


import xylo
import xylo.types as t
import xylo.woods
import xylo.cut

import xylo.opt

import xylo.tuning
import xylo.database

tuning = xylo.tuning.yamaha_YX500R

for note in list(range(45, 57)) + [75, 77, 80]:
  for num_dims in range(3,6):
    options = xylo.opt.Options(num_generations = 10, strategy = evosax.OpenES(popsize = 2000, num_dims = num_dims), absolute_tolerance =  1e-3)
    db = xylo.database.get()
    best = db.get_best_for_dims(note, num_dims)
    options_ = options._replace(init_mean = best)

    bar = db.get_bar(note)
    wood = db.get_wood(note)

    fundamental = tuning.note_to_freq(note)
    partials, weights = tuning.note_to_weights(note)
    sol = xylo.opt.optimize_geometry(bar, wood, options_, fundamental, jnp.array(partials), jnp.array(weights))
    spline = sol.best_member

    xylo.database.with_db(lambda db: db.set_best_for_dims(note, num_dims, sol.best_member.tolist(), sol.best_fitness.tolist()))

    init_mean = sol.best_member

    sections = xylo.cut.spline(bar, spline)
    swp = xylo.sweep.sweep(wood, bar, sections, t.sweep_default)

    print(note)
    print(f"--------------- (loss {sol.best_fitness})")
    print(swp.harmonics / fundamental)
    print("")

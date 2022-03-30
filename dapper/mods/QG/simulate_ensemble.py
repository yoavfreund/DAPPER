"""Ensemble generation for stream function time series for QG (quasi-geostrophic) model."""

import numpy as np
import dapper.mods as modelling
import dapper as dpr
from dapper.mods.QG import square, model_config, default_prms, shape
import dapper.tools.progressbar as pb

###########
# Auxiliary plotting function
###########
def show(x0, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    im = ax.imshow(square(x0))
    im.set_clim(-30, 30)

    def update(x):
        im.set_data(square(x))
    return update


#########################
# Free ensemble run
#########################
def gen_ensemble_sample(model, nSamples, nEnsemble, SpinUp, Spacing):
    simulator = modelling.with_recursion(model.step, prog="Simulating")
    K         = SpinUp + nSamples*Spacing
    Nx        = np.prod(shape)  # total state length
    init      = np.random.normal(loc=0.0, scale=0.1, size=[nEnsemble, Nx])
    sample    = simulator(init, K, 0.0, model.prms["dtout"])
    return sample[SpinUp::Spacing,:, :]


###########
# Main
###########
# Generate time-series data of a simulated state and obs:
np.random.seed(123)
Ne = 10
model_config.mp = True
fname = "QG-ts-en-" + str(Ne) + ".npz"
fname = dpr.rc.dirs.data / fname
sample = gen_ensemble_sample(model_config("sample_generation", {}), 
                         400, Ne, 10, 10)

np.savez(fname, ens=sample)

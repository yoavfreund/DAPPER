# ## Illustrate usage of DAPPER to (interactively) run a synthetic ("twin") experiment.

# #### Imports
import numpy as np
import dapper as dpr
import dapper.da_methods as da

# #### Load experiment setup: the hidden Markov model (HMM)

from dapper.mods.QG.sakov2008 import HMM

# #### Generate the same random numbers each time this script is run

seed = dpr.set_seed(3000)

# #### Simulate synthetic truth (xx) and noisy obs (yy)

HMM.tseq.T = 200
xx, yy = HMM.simulate()

# #### Specify a DA method configuration ("xp" is short for "experiment")

xp_EnKF = da.EnKF('Sqrt', N=150, infl=1.02, rot=True)

# #### Assimilate yy, knowing the HMM; xx is used to assess the performance

xp_EnKF.assimilate(HMM, xx, yy)

# #### Average the time series of various statistics
xp_EnKF.stats.average_in_time()

# #### Print some averages

print(xp_EnKF.avrgs.tabulate(['rmse.a', 'rmv.a']))

# #### save the time series

fname = "QG-DA-cycle-ts.npz"
fname = dpr.rc.dirs.data / fname
np.savez(fname, 
         xx = xx[1:, :],
         yy = yy,
         EnKF_analysis=xp_EnKF.a_time_series,
         EnKF_forecast=xp_EnKF.f_time_series,
         EnKF_analysis_rms = xp_EnKF.stats.err.rms.a,
         EnKF_forecast_rms = xp_EnKF.stats.err.rms.f,
         EnKF_analysis_spread = xp_EnKF.stats.spread.m.a,
         EnKF_forecast_spread = xp_EnKF.stats.spread.m.f,
         )

import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import itertools as it
import numpy.random as npr


def sos(test_trace):
    return np.sum((expt_trace-test_trace)**2)


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular) (not available atm)
# 8. Gokhale 2017 ex293 (not available atm)

protocol = 1
model_number = 3
noise_sd = 0.5

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)
solve_end = 400
times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
original_gs = np.array(original_gs)

# set up APSimulator --- there are more options available, see CMA-ES/MCMC codes
ap = ap_simulator.APSimulator()
ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
ap.DefineModel(model_number)



#ICs = ap.GetStateVariables()
ap.SetToModelInitialConditions()
expt_trace = ap.SolveForVoltageTraceWithParams(original_gs) + noise_sd*npr.randn(len(times))

# reset ICs before each solve, normally I put this inside of a newly-defined solve function so it always happens
ap.SetToModelInitialConditions()
test_trace = ap.SolveForVoltageTraceWithParams(1.5*original_gs)

example_sos = sos(test_trace)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(times, expt_trace, label='expt')
ax.plot(times, test_trace, label='test')
ax.set_title(model_name)
ax.legend(loc="best")
plt.show(block=True)


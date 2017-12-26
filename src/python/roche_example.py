import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import itertools as it
import numpy.random as npr


def example_likelihood_function(trace):
    return np.sum(trace**2)
    
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


roche_file = "projects/PyAP/python/input/roche_ten_tusscher/traces/Trace_2_2_100_1.csv"
expt_times, expt_trace = np.loadtxt(roche_file, delimiter=',').T

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)

solve_start = expt_times[0]
solve_end = expt_times[-1]
solve_timestep = expt_times[1]-expt_times[0]
stimulus_magnitude = -50  # wrong
stimulus_duration = 2
stimulus_period = 1000
stimulus_start_time = 50

model_number = 4

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(expt_times, expt_trace)

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
original_gs = np.array(original_gs)

ap = ap_simulator.APSimulator()
ap.DefineStimulus(stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time)
ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
ap.DefineModel(model_number)

ap.SetToModelInitialConditions()
temp_gs = np.copy(original_gs)
test_trace = ap.SolveForVoltageTraceWithParams(temp_gs)
ax.plot(expt_times, test_trace)


ap.SetToModelInitialConditions()
ap.SetMembraneCapacitance(0.00005631)  # 56.31 pF
temp_gs = np.copy(original_gs)
test_trace = ap.SolveForVoltageTraceWithParams(temp_gs)
ax.plot(expt_times, test_trace)

ax.set_title(model_name)
plt.show(block=True)





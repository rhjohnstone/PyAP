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




fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.grid()
first_trace = 100
num_traces = 16
for i in xrange(num_traces):
    t = first_trace + i
    roche_file = "projects/PyAP/python/input/roche_ten_tusscher/traces/Trace_2_2_{}_1.csv".format(t)
    expt_times, expt_trace = np.loadtxt(roche_file, delimiter=',').T
    if i==0:
        num_pts = len(expt_times)
        all_expts = np.zeros((num_traces, num_pts))
    all_expts[i, :] = expt_trace
    ax.plot(expt_times, expt_trace, color='blue')
    
triangle_t0 = 50.2
triangle_t1 = 51.2
triangle_idx = (triangle_t0 < expt_times) & (expt_times < triangle_t1)
triangle_times = expt_times[triangle_idx]
triangle_Vs = all_expts[:, triangle_idx]

m, c = np.polyfit(triangle_times, triangle_Vs.mean(axis=0), deg=1)

Cm = 56.31  # 56.31 pF
I_stim = -m*Cm
print "I_stim = {} pA".format(I_stim)

fitted_V = m*triangle_times + c
    
dc_on = 50
dc_off = 51.3
ax.set_xlim(50, 52)
ax.axvline(dc_on, color='red', lw=2, clip_on=False)
ax.axvline(dc_off, color='red', lw=2)
ax.set_ylabel("Membrane voltage (mV)")
ax.set_xlabel("Time (ms)")

ax.plot(triangle_times, fitted_V, color='red')
ax.plot(triangle_times[0], fitted_V[0], 'x', color='red', ms=10, mew=2, zorder=10)
ax.plot(triangle_times[-1], fitted_V[-1], 'x', color='red', ms=10, mew=2, zorder=10)

fig.tight_layout()




plt.show()
sys.exit()

solve_start = expt_times[0]
solve_end = expt_times[-1]
solve_timestep = expt_times[1]-expt_times[0]
stimulus_magnitude = -50  # wrong
stimulus_duration = 2
stimulus_period = 1000
stimulus_start_time = 50

model_number = 4


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





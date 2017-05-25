import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys


def example_likelihood_function(trace):
    return np.sum(trace**2)


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293

model_number = 4
protocol = 1

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)

original_gs, g_parameters = ps.get_original_params(model_number)

times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

ap = ap_simulator.APSimulator()

ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
ap.DefineModel(model_number)
try:
    true_trace = ap.SolveForVoltageTraceWithParams(original_gs)
except ap_simulator.CPPException as e:
    print e.GetMessage
    sys.exit()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times,true_trace)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane voltage (mV)")
ax.set_title("Model {}".format(model_number))
plt.show(block=True)

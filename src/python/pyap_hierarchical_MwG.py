import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
    
    
def dog_trace_path(trace_number):
    return "projects/PyAP/python/input/dog_teun_csv/dog_AP_trace_{}.csv".format(trace_number)


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293

model_number = 6
protocol = 1

trace_numbers = [100, 101]
expt_traces = []
for i, t in enumerate(trace_numbers):
    trace_path = dog_trace_path(t)
    if i==0:
        expt_times, trace = 1000*np.loadtxt(trace_path,delimiter=',').T
    else:
        trace = 1000*np.loadtxt(trace_path,delimiter=',',usecols=[1])
    expt_traces.append(np.copy(trace))
aps = []
model_traces = []

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)

original_gs, g_parameters = ps.get_original_params(model_number)

times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

"""for i, t in enumerate(trace_numbers):
    aps.append(ap_simulator.APSimulator())
    aps[i].DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    aps[i].DefineSolveTimes(solve_start,solve_end,solve_timestep)
    aps[i].DefineModel(model_number)
    try:
        model_traces.append(aps[i].SolveForVoltageTraceWithParams(original_gs*(1.+0.1*i)))
    except ap_simulator.CPPException as e:
        print e.GetMessage
        sys.exit()"""
        
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
for i, t in enumerate(trace_numbers):
    ax.plot(expt_times,expt_traces[i], label="Trace {}".format(t))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane voltage (mV)")
ax.set_title("Model {}".format(model_number))
ax.legend()
fig.tight_layout()
plt.show(block=True)

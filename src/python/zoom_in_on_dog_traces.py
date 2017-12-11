#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ap_simulator
import numpy as np
import time
import pyap_setup as ps
import sys


def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)


def solve_for_voltage_trace_without_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)
        
        
data_clamp_on = 9.875
data_clamp_off = 11.875

zoomed_xlim = (9, 12.5)



ax_y = 3
lw = 1
fig, axs = plt.subplots(1, 2, figsize=(2*ax_y,ax_y))
for i in xrange(2):
    axs[i].set_ylabel('Membrane voltage (mV)')
    axs[i].set_xlabel('Time (ms)')
    axs[i].grid()

num_traces = 2
for i in xrange(num_traces):
    trace_number = 150 + i
    trace_path = "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_{}.csv".format(trace_number)

    expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
    num_pts = len(expt_trace)
    
    axs[0].plot(expt_times, expt_trace)
    
    zoomed_where = (zoomed_xlim[0] <= expt_times) && (expt_times <= zoomed_xlim[1])
    axs[1].plot(expt_times[zoomed_where], expt_trace[zoomed_where])

cs = ['#1b9e77','#d95f02','#7570b3']

fig.tight_layout()
fig_file = "dog_traces_zoomed.png"
print fig_file
plt.show()




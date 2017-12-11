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

data_clamp_time = [data_clamp_on, data_clamp_off]

zoomed_xlim = (9, 12.5)
zoomed_ylim = (-100, 0)

ax_y = 3
lw = 1
fig, axs = plt.subplots(1, 2, figsize=(3*ax_y,ax_y))
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
    
    for j in xrange(2):
        axs[j].plot(expt_times, expt_trace)
    axs[1].set_xlim(zoomed_xlim)
    axs[1].set_ylim(zoomed_ylim)
    
    zoomed_where = (zoomed_xlim[0] <= expt_times) & (expt_times <= zoomed_xlim[1])

rectangle_y = expt_trace[np.where(expt_times==data_clamp_off)]
axs[1].fill_between(data_clamp_time, [rectangle_y, rectangle_y], color='lightgray')

fig.tight_layout()
fig_file = "dog_traces_zoomed.png"
print fig_file
plt.show()




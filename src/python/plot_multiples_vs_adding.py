import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys


def example_likelihood_function(trace):
    return np.sum(trace**2)
    

def solve_with_params(temp_params):
    try:
        ap.SetToModelInitialConditions()
        return ap.SolveForVoltageTraceWithParams(temp_params)
    except ap_simulator.CPPException as e:
        print e.GetMessage
        sys.exit()

# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular) (not available atm)
# 8. Gokhale 2017 ex293 (not available atm)

protocol = 1

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)
for model_number in xrange(6,7):

    if model_number==1:
        solve_end = 60
    else:
        solve_end = 400
    times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

    k = 2.
    G_CaL_scales = [k, 1./k]

    original_gs, g_parameters, model_name = ps.get_original_params(model_number)
    original_gs = np.array(original_gs)

    ap = ap_simulator.APSimulator()
    ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap.DefineModel(model_number)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5),sharey=True)
    
    original_trace = solve_with_params(original_gs)
    for ax in [ax1, ax2]:
        ax.grid()
        ax.plot(times, original_trace, color='red', label='Original')
    
    for m in [0.5, 2]:
        scaled_gs = np.copy(original_gs)
        scaled_gs[1] *= m
        trace = solve_with_params(scaled_gs)
        ax2.plot(times, trace, label=r"$G_{CaL} \times "+str(m)+"$")
        
    scaled_gs = np.copy(original_gs)
    scaled_gs[1] *= 0.5
    trace = solve_with_params(scaled_gs)
    ax1.plot(times, trace, label=r"$G_{CaL} - 50\%$")
        
    scaled_gs = np.copy(original_gs)
    scaled_gs[1] *= 1.5
    trace = solve_with_params(scaled_gs)
    ax1.plot(times, trace, label=r"$G_{CaL} + 50\%$")
        
        
    ax2.set_xlabel("Time (ms)")
    ax1.set_xlabel("Time (ms)")
    ax2.set_ylabel("Membrane voltage (mV)")
    #ax1.set_title(model_name)
    ax1.legend()
    ax2.legend()
    plt.show(block=True)


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
# 7. Paci (SC-CM ventricular) (not available atm)
# 8. Gokhale 2017 ex293 (not available atm)

protocol = 1

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)
for model_number in xrange(1,3):

    if i==1:
        solve_end = 100
    else:
        solve_end = 400
    times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

    multiples = [0.3, 1, 3, 10]

    original_gs, g_parameters, model_name = ps.get_original_params(model_number)
    original_gs = np.array(original_gs)

    ap = ap_simulator.APSimulator()
    ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap.DefineModel(model_number)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    
    for m in multiples:
        scaled_gs = m * original_gs
        try:
            ap.SetToModelInitialConditions()
            trace = ap.SolveForVoltageTraceWithParams(scaled_gs)
        except ap_simulator.CPPException as e:
            print e.GetMessage
            sys.exit()
        ax.plot(times, trace, label=m)
        
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane voltage (mV)")
    ax.set_title(model_name)
    ax.legend()
    plt.show(block=True)


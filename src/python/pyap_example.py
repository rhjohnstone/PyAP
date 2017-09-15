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


solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)
for model_number in xrange(3,4):

    solve_end = 400
    times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

    original_gs, g_parameters, model_name = ps.get_original_params(model_number)
    original_gs = np.array(original_gs)

    ap = ap_simulator.APSimulator()
    ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap.DefineModel(model_number)
    
    ap.SetToModelInitialConditions()
    expt_trace = ap.SolveForVoltageTraceWithParams(original_gs)
    print "expt_trace original_gs done"
    
    p_size = 0.05
    perturbations = [p_size*np.ones(len(original_gs)), -p_size*np.ones(len(original_gs))]
    sosses = []
    how_many_perturbs = 10
    while len(perturbations) < how_many_perturbs:
        temp_p = p_size*(npr.randint(0,2,len(original_gs))*2.-1.)
        if not any((temp_p == x).all() for x in perturbations):
            perturbations.append(temp_p)
    print perturbations
    
    
    #ax.grid()
    
    for p in perturbations:
        scaled_gs = (1.+p) * original_gs
        try:
            ap.SetToModelInitialConditions()
            trace = ap.SolveForVoltageTraceWithParams(scaled_gs)
        except ap_simulator.CPPException as e:
            print e.GetMessage
            sys.exit()
        print "\n", (1.+p)
        ss = sos(trace)
        print ss
        sosses.append(ss)
        #ax.plot(times, trace, label=m)
    #ax.set_ylabel("Membrane voltage (mV)")
    #axs.set_xlabel("Time (ms)")
    
    #ax.set_title(model_name)
    #ax.legend()
    
    print sosses
    
    print "Min sos: {} from {}\n".format(np.min(sosses), perturbations[np.argmin(sosses)])
    
    #fig.tight_layout()
    #plt.show(block=True)


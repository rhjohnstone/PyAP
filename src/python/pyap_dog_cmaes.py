import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import cma


def solve_for_voltage_trace(temp_g_params):
    ap_model.SetToModelInitialConditions()
    return ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    
    
def obj(temp_test_params):
    if np.any(temp_test_params < 0):
        negs = temp_test_params[np.where(temp_test_params<0)]
        return 1e9 * (1 - np.sum(negs))
    temp_test_trace = solve_for_voltage_trace(temp_test_params)
    return np.sum((temp_test_trace-expt_trace)**2)


def run_cmaes(cma_index):
    opts = cma.CMAOptions()
    opts['seed'] = cma_index
    x0 = original_gs
    sigma0 = 0.000001
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        es.tell(X, [obj(x) for x in X])
        es.disp()
    res = es.result()
    best_params = res[0]
    best_obj = res[1]
    return best_params, best_obj

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
extra_K_conc = 5.4
trace_numbers = [100]#, 101]
num_solves = 1

expt_traces = []
for i, t in enumerate(trace_numbers):
    trace_path = ps.dog_trace_path(t)
    if i==0:
        expt_times, trace = 1000*np.loadtxt(trace_path,delimiter=',').T
    else:
        trace = 1000*np.loadtxt(trace_path,delimiter=',',usecols=[1])
    expt_traces.append(np.copy(trace))
aps = []
model_traces = []

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)

stimulus_duration = 2.5
solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]
stimulus_start_time = 9.625
how_many_cmaes_runs = 2

original_gs, g_parameters = ps.get_original_params(model_number)

cmaes_indices = range(how_many_cmaes_runs)


figs = []
for i, t in enumerate(trace_numbers):
    plt.close()
    expt_trace = expt_traces[i]
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap_model.DefineModel(model_number)
    ap_model.SetExtracellularPotassiumConc(extra_K_conc)
    ap_model.SetNumberOfSolves(num_solves)
    best_paramses = []
    best_fs = []
    figs.append(plt.figure())
    ax = figs[i].add_subplot(111)
    ax.grid()
    ax.set_title('Trace {}'.format(t))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane voltage (mV)')
    ax.plot(expt_times, expt_traces[i], label="Expt")
    for j in cmaes_indices:
        best_params, best_f = run_cmaes(j)
        best_paramses.append(best_params)
        best_fs.append(best_f)
        ax.plot(expt_times, solve_for_voltage_trace(best_paramses[j]), label="Best f = {}".format(round(best_fs[j],2)))
    ax.legend()
    figs[i].tight_layout()
plt.show(block=True)
    

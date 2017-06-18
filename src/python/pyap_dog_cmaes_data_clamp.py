import ap_simulator
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import cma
import multiprocessing as mp


def solve_for_voltage_trace(temp_g_params, ap_model):
    ap_model.SetToModelInitialConditions()
    return ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    
    
def obj(temp_test_params, ap_model):
    if np.any(temp_test_params < 0):
        negs = temp_test_params[np.where(temp_test_params<0)]
        return 1e9 * (1 - np.sum(negs))
    temp_test_trace = solve_for_voltage_trace(temp_test_params, ap_model)
    return np.sum((temp_test_trace-expt_trace)**2)


def run_cmaes(cma_index):
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap_model.DefineModel(model_number)
    ap_model.SetExtracellularPotassiumConc(extra_K_conc)
    ap_model.SetNumberOfSolves(num_solves)
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
    #npr.seed(cma_index)
    #opts = cma.CMAOptions()
    #npr.seed(cma_index)
    #opts['seed'] = cma_index
    #options = {'seed':cma_index}
    x0 = original_gs * (1. + 0.001*npr.randn(num_params))
    figz = plt.figure()
    axz = figz.add_subplot(111)
    trace0 = solve_for_voltage_trace(x0, ap_model)
    f0 = obj(x0, ap_model)
    axz.plot(expt_times, expt_trace, label='Expt')
    axz.plot(expt_times, trace0, label='f0 = {}'.format(round(f0,2)))
    axz.set_xlabel('Time (ms)')
    axz.set_ylabel('Membrane voltage (mV)')
    axz.legend()
    axz.grid()
    #figz.tight_layout()
    figz.savefig(cmaes_dir+'trace_{}_cmaes_index_{}_initial_fit.png'.format(t,cma_index))
    plt.close()
    print "x0:", x0
    obj0 = obj(x0, ap_model)
    print "obj0:", round(obj0, 2)
    x0[np.where(x0<0)] = 1e-9
    sigma0 = 0.000001
    es = cma.CMAEvolutionStrategy(x0, sigma0)#, options)
    while not es.stop():
        X = es.ask()
        es.tell(X, [obj(x, ap_model) for x in X])
        es.disp()
    res = es.result()
    answer = np.concatenate((res[0],[res[1]]))
    return answer


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293



num_cores = 2  # make 16 for ARCUS-B!!

model_number = 6
protocol = 1
extra_K_conc = 5.4
trace_numbers = [100]#, 101]
num_solves = 2
use_data_clamp = True

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

data_clamp_on = expt_times[39]
data_clamp_off = expt_times[47]
print data_clamp_on, data_clamp_off

solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]
stimulus_magnitude = 0
stimulus_duration = 1
stimulus_period = 1000
stimulus_start_time = 0.
original_gs, g_parameters = ps.get_original_params(model_number)
num_params = len(original_gs)

how_many_cmaes_runs = 2
cmaes_indices = range(how_many_cmaes_runs)


figs = []
for i, t in enumerate(trace_numbers):
    cmaes_dir, best_fit_file = ps.dog_cmaes_path(t)
    expt_trace = expt_traces[i]
    best_paramses = []
    best_fs = []
    best_both = []
    figs.append(plt.figure())
    ax = figs[i].add_subplot(111)
    ax.grid()
    ax.set_title('Trace {}'.format(t))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane voltage (mV)')
    ax.plot(expt_times, expt_traces[i], label="Expt")
    pool = mp.Pool(num_cores)
    best_boths = pool.map_async(run_cmaes, cmaes_indices).get(999999999)
    pool.close()
    pool.join()
    best_boths = np.array(best_boths)
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap_model.DefineModel(model_number)
    ap_model.SetExtracellularPotassiumConc(extra_K_conc)
    ap_model.SetNumberOfSolves(num_solves)
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_traces[i])
    for j in cmaes_indices:
        #best_params, best_f = run_cmaes(j)
        best_params = best_boths[j,:-1]
        best_f = best_boths[j,-1]
        ax.plot(expt_times, solve_for_voltage_trace(best_params, ap_model), label="Best f = {}".format(round(best_f,2)))
    ax.legend()
    figs[i].tight_layout()
    np.savetxt(best_fit_file, best_boths)
    figs[i].savefig(cmaes_dir+'trace_{}_best_fits.png'.format(t))
    plt.close()
#plt.show(block=True)
    

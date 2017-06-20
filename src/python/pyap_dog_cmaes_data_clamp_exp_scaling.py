import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ap_simulator
import numpy as np
import numpy.random as npr
import time
import pyap_setup as ps
import sys
import cma
import multiprocessing as mp


def exponential_scaling(unscaled_params):
    return original_gs ** (unscaled_params/10.)


def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    
    
def obj(temp_test_params, temp_ap_model):
    scaled_params = exponential_scaling(temp_test_params)
    temp_test_trace = solve_for_voltage_trace(scaled_params, temp_ap_model)
    return np.sum((temp_test_trace-expt_trace)**2)


def run_cmaes(cma_index):
    start = time.time()
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap_model.DefineModel(model_number)
    ap_model.SetExtracellularPotassiumConc(extra_K_conc)
    ap_model.SetIntracellularPotassiumConc(intra_K_conc)
    ap_model.SetExtracellularSodiumConc(extra_Na_conc)
    ap_model.SetIntracellularSodiumConc(intra_Na_conc)
    ap_model.SetNumberOfSolves(num_solves)
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
    npr.seed(cma_index)  # can't fix CMA-ES seed for some reason
    #opts = cma.CMAOptions()
    #npr.seed(cma_index)
    #opts['seed'] = cma_index
    #options = {'seed':cma_index}
    x0 = 10. + npr.randn(num_params)
    print "x0:", x0
    obj0 = obj(x0, ap_model)
    print "obj0:", round(obj0, 2)
    sigma0 = 0.1
    es = cma.CMAEvolutionStrategy(x0, sigma0)#, options)
    while not es.stop():
        X = es.ask()
        es.tell(X, [obj(x, ap_model) for x in X])
        es.disp()
    res = es.result()
    answer = np.concatenate((exponential_scaling(res[0]),[res[1]]))
    time_taken = time.time()-start
    print "\n\nTime taken by one CMA-ES run: {} s\n\n".format(round(time_taken))
    return answer


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293



num_cores = 3  # make 16 for ARCUS-B!!

model_number = 6
protocol = 1
extra_K_conc = 5.4
intra_K_conc = 130
extra_Na_conc = 140
intra_Na_conc = 10
trace_numbers = range(150,300)#, 101]

num_solves = 2

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

data_clamp_on = expt_times[40]
data_clamp_off = expt_times[48]
print data_clamp_on, data_clamp_off

solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]
stimulus_magnitude = 0
stimulus_duration = 1
stimulus_period = 1000
stimulus_start_time = 0.
original_gs, g_parameters = ps.get_original_params(model_number)
num_params = len(original_gs)

how_many_cmaes_runs = 6
cmaes_indices = range(how_many_cmaes_runs)

plt.close()
for i, t in enumerate(trace_numbers):
    cmaes_dir, best_fit_file = ps.dog_cmaes_path(t)
    expt_trace = expt_traces[i]
    best_paramses = []
    best_fs = []
    best_both = []
    try:
        pool = mp.Pool(num_cores)
        best_boths = pool.map_async(run_cmaes, cmaes_indices).get(9999)
        pool.close()
        pool.join()
    except:
        continue
    
    best_boths = np.array(best_boths)
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap_model.DefineModel(model_number)
    ap_model.SetExtracellularPotassiumConc(extra_K_conc)
    ap_model.SetIntracellularPotassiumConc(intra_K_conc)
    ap_model.SetExtracellularSodiumConc(extra_Na_conc)
    ap_model.SetIntracellularSodiumConc(intra_Na_conc)
    ap_model.SetNumberOfSolves(num_solves)
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_traces[i])
    best_fit_index = np.argmin(best_boths[:,-1])
    best_params = best_boths[best_fit_index,:-1]
    best_f = best_boths[best_fit_index,-1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title('Trace {}'.format(t))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane voltage (mV)')
    ax.plot(expt_times, expt_traces[i], label="Expt")
    ax.plot(expt_times, solve_for_voltage_trace(best_params, ap_model), label="Best f = {}".format(round(best_f,2)))
    ax.legend()
    fig.tight_layout()
    np.savetxt(best_fit_file, best_boths)
    fig.savefig(cmaes_dir+'trace_{}_best_fits.png'.format(t))
    fig.savefig(cmaes_dir+'trace_{}_best_fits.svg'.format(t))
    plt.close()
    

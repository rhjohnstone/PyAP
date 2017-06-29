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
import shutil
import os


def exponential_scaling(unscaled_params):
    return original_gs ** (unscaled_params/10.)


def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    try:
        return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    except ap_simulator.CPPException, e:
        print e.GetShortMessage
        sys.exit()
    
    
def obj(temp_test_params, temp_ap_model):
    scaled_params = exponential_scaling(temp_test_params)
    temp_test_trace = solve_for_voltage_trace(scaled_params, temp_ap_model)
    return np.sum((temp_test_trace-expt_trace)**2)


def run_cmaes(cma_index):
    print "\n\n\n{}, cma {}\n\n\n".format(ps.trace_name, cma_index)
    start = time.time()
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, ps.pyap_options["stimulus_period"], stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start, solve_end, solve_timestep)
    ap_model.DefineModel(ps.pyap_options["model_number"])
    ap_model.SetExtracellularPotassiumConc(ps.pyap_options["extra_K_conc"])
    ap_model.SetIntracellularPotassiumConc(ps.pyap_options["intra_K_conc"])
    ap_model.SetExtracellularSodiumConc(ps.pyap_options["extra_Na_conc"])
    ap_model.SetIntracellularSodiumConc(ps.pyap_options["intra_Na_conc"])
    ap_model.SetNumberOfSolves(ps.pyap_options["num_solves"])
    ap_model.UseDataClamp(ps.pyap_options["data_clamp_on"], ps.pyap_options["data_clamp_off"])
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
# 9. Davies (canine) linearised by RJ
# 10. Paci linearised by RJ

all_time_start = time.time()

num_cores = 16  # make 16 for ARCUS-B!!

expt_times, expt_trace = np.loadtxt(ps.trace_path,delimiter=',').T

solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]
stimulus_magnitude = 0.
stimulus_duration = 1.
stimulus_start_time = 0.
original_gs, g_parameters = ps.get_original_params(ps.pyap_options["model_number"])
num_params = len(original_gs)

how_many_cmaes_runs = 1
cmaes_indices = range(how_many_cmaes_runs)

trace_start_time = time.time()
cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(ps.pyap_options["model_number"])

if num_cores > 1:
    pool = mp.Pool(num_cores)
    best_boths = pool.map_async(run_cmaes, cmaes_indices).get(99999)
    pool.close()
    pool.join()
else:
    print "not parallel"
    best_boths = []
    for ci in cmaes_indices:
        best_boths.append(run_cmaes(ci))
best_boths = np.array(best_boths)
np.savetxt(cmaes_best_fits_file, best_boths)
ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, ps.pyap_options["stimulus_period"], stimulus_start_time)
ap_model.DefineSolveTimes(solve_start, solve_end, solve_timestep)
ap_model.DefineModel(ps.pyap_options["model_number"])
ap_model.SetExtracellularPotassiumConc(ps.pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(ps.pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(ps.pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(ps.pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(ps.pyap_options["num_solves"])
ap_model.UseDataClamp(ps.pyap_options["data_clamp_on"], ps.pyap_options["data_clamp_off"])
ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
best_fit_index = np.argmin(best_boths[:,-1])
best_params = best_boths[best_fit_index,:-1]
best_f = best_boths[best_fit_index,-1]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_title(ps.trace_name)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane voltage (mV)')
ax.plot(expt_times, expt_trace, label="Expt")
ax.plot(expt_times, solve_for_voltage_trace(best_params, ap_model), label="Best f = {}".format(round(best_f,2)))
ax.legend()
fig.tight_layout()
fig.savefig(best_fit_png)
fig.savefig(best_fit_svg)
plt.close()
trace_time_taken = time.time()-trace_start_time
print "\n\n{}, time taken: {} s = {} min\n\n".format(ps.trace_name, round(trace_time_taken), round(trace_time_taken/60.))


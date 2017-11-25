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
import argparse

npexp = np.exp
nplog = np.log
npinf = np.inf
npsum = np.sum
npsqrt = np.sqrt

sigma_uniform_lower = 1e-3
sigma_uniform_upper = 25.
true_noise_sd = 0.5
omega = 0.5*nplog(10)  # s.d. of Normal priors on lnGs
two_omega_sq = 2.*omega**2
        

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
requiredNamed.add_argument("--num-cores", type=int, help="how many cores to run multiple CMA-ES minimisations on", required=True)
requiredNamed.add_argument("--num-runs", type=int, help="how many CMA-ES minimisations to run", required=True)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"

pyap_options = {}
with open(options_file, 'r') as infile:
    for line in infile:
        (key, val) = line.split()
        if (key == "model_number") or (key == "num_solves"):
            val = int(val)
        else:
            val = float(val)
        pyap_options[key] = val
        
data_clamp_on = pyap_options["data_clamp_on"]
data_clamp_off = pyap_options["data_clamp_off"]

def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))


def solve_for_voltage_trace_without_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        sys.exit()


if data_clamp_on < data_clamp_off:
    solve_for_voltage_trace = solve_for_voltage_trace_with_initial_V
    print "Solving after setting V(0) = data(0)"
else:
    solve_for_voltage_trace = solve_for_voltage_trace_without_initial_V
    print "Solving without setting V(0) = data(0)"
    
    
def log_target(temp_params, ap_model, expt_trace):
    """Log target distribution with Normal prior for lnGs, uniform for sigma"""
    temp_lnGs, temp_sigma = temp_params[:-1], temp_params[-1]
    if not (sigma_uniform_lower < temp_sigma < sigma_uniform_upper):
        return -npinf
    else:
        try:
            test_trace = solve_for_voltage_trace(temp_lnGs, ap_model, expt_trace)
            return -num_pts*nplog(temp_sigma) - npsum((test_trace-expt_trace)**2)/(2.*temp_sigma**2) - npsum((temp_lnGs-log_gs)**2)/two_omega_sq

        except:
            #print "Failed to solve at iteration", t
            print "exp(temp_lnGs):\n", npexp(temp_lnGs)
            print "original_gs:\n", original_gs
            return -npinf



def run_cmaes(cma_index):
    print "\n\n\n{}, cma {}\n\n\n".format(trace_name, cma_index)
    start = time.time()
    ap_model = ap_simulator.APSimulator()
    if (data_clamp_on < data_clamp_off):
        ap_model.DefineStimulus(0, 1, 1000, 0)  # no injected stimulus current
        ap_model.DefineModel(pyap_options["model_number"])
        ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
        ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
    else:
        ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
        ap_model.DefineModel(pyap_options["model_number"])
    ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
    ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
    ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
    ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
    ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
    ap_model.SetNumberOfSolves(pyap_options["num_solves"])
    npr.seed(cma_index)  # can't fix CMA-ES seed for some reason
    #opts = cma.CMAOptions()
    #npr.seed(cma_index)
    #opts['seed'] = cma_index
    #options = {'seed':cma_index}
    x0 = np.concatenate((nplog(original_gs), [true_noise_sd]))  + 0.5*npr.randn(num_params)
    sigma0 = 0.1
    print "x0:", x0
    print "sigma0:", sigma0
    obj0 = obj(x0, ap_model)
    print "obj0:", round(obj0, 2)
    es = cma.CMAEvolutionStrategy(x0, sigma0)#, options)
    while not es.stop():
        X = es.ask()
        #for q in X:
        #    print X
        es.tell(X, [log_target(x, ap_model, expt_trace) for x in X])
        es.disp()
    res = es.result()
    answer = np.concatenate((npexp(res[0]),[res[1]]))
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

phi = 1.61803398875

all_time_start = time.time()

num_cores = args.num_cores  # make 16 for ARCUS-B!!

expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)
if pyap_options["model_number"] == 1:
    solve_end = 100  # just for HH
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)
num_params = num_gs + 1

how_many_cmaes_runs = args.num_runs
cmaes_indices = range(how_many_cmaes_runs)

trace_start_time = time.time()
cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files_lnG(pyap_options["model_number"], expt_name, trace_name)

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
if (data_clamp_on < data_clamp_off):
    ap_model.DefineStimulus(0, 1, 1000, 0)  # no injected stimulus current
    ap_model.DefineModel(pyap_options["model_number"])
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
else:
    ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
    ap_model.DefineModel(pyap_options["model_number"])
ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])
if (data_clamp_on < data_clamp_off):
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
best_fit_index = np.argmin(best_boths[:,-1])
best_params = best_boths[best_fit_index,:-1]
print "best_params:\n", best_params
best_f = best_boths[best_fit_index,-1]
fig = plt.figure(figsize=(phi*4,4))
ax = fig.add_subplot(111)
ax.grid()
ax.set_title(model_name)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane voltage (mV)')
ax.plot(expt_times, expt_trace, label=trace_name)
ax.plot(expt_times, solve_for_voltage_trace(best_params, ap_model), label="Best f = {}".format(int(best_f)))
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig(best_fit_png)
fig.savefig(best_fit_svg)
best_fit_pdf = best_fit_png[:-3]+"pdf"
fig.savefig(best_fit_pdf)
plt.close()
trace_time_taken = time.time()-trace_start_time
print "\n\n{}, time taken: {} s = {} min\n\n".format(trace_name, round(trace_time_taken), round(trace_time_taken/60.))


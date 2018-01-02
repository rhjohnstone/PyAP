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

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)
log_gs = nplog(original_gs)

# convert Cm and Istim to correct units, model-specific
if pyap_options["model_number"]==3:  # LR
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] * 1e-6
elif pyap_options["model_number"]==4:  # TT
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
    indices_to_keep = [0, 1, 2, 3, 4, 6]  # which currents we are fitting
elif pyap_options["model_number"]==5:  # OH
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
elif pyap_options["model_number"]==7:  # Pa
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-12
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-12
    indices_to_keep = [0, 1, 2, 3, 4, 9, 11]  # which currents we are fitting
num_params_to_fit = len(indices_to_keep)


temp_Gs = np.zeros(num_gs)
def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    
    temp_Gs[indices_to_keep] = npexp(temp_lnG_params)
    try:
        return ap_model.SolveForVoltageTraceWithParams(temp_Gs)
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)


if data_clamp_on < data_clamp_off:
    solve_for_voltage_trace = solve_for_voltage_trace_with_initial_V
    print "Solving after setting V(0) = data(0)"
else:
    sys.exit("This is just for Roche, so there should be some data-clamp.")
    
    
def sum_of_squares(temp_lnGs, ap_model, expt_trace):
    try:
        test_trace = solve_for_voltage_trace(temp_lnGs, ap_model, expt_trace)
        return np.sum((test_trace-expt_trace)**2)
    except Exception as e:
        print e
        #print "Failed to solve at iteration", t
        print "exp(temp_lnGs):\n", npexp(temp_lnGs)
        print "original_gs:\n", original_gs
        return npinf


def compute_initial_sigma(sos):
    return np.sqrt((1.*sos)/num_pts)
    
    
def log_likelihood(sos, sigma):
    return -num_pts*nplog(sigma) - sos/(2.*sigma**2)


def run_cmaes(cma_index):
    print "\n\n\n{}, cma {}\n\n\n".format(trace_name, cma_index)
    start = time.time()
    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude, pyap_options["stimulus_duration_ms"], pyap_options["stimulus_period_ms"], pyap_options["stimulus_start_ms"])
    ap_model.DefineModel(pyap_options["model_number"])
    if (data_clamp_on < data_clamp_off):
        ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
        ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)        
    ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
    ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
    ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
    ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
    ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
    ap_model.SetNumberOfSolves(pyap_options["num_solves"])
    ap_model.SetMembraneCapacitance(Cm)
    npr.seed(cma_index)  # can't fix CMA-ES seed for some reason
    opts = cma.CMAOptions()
    print opts["tolfun"] 
    #opts["tolfun"] = 1e-4
    #npr.seed(cma_index)
    #opts['seed'] = cma_index
    #options = {'seed':cma_index}
    x0 = nplog(original_gs[indices_to_keep])  + 0.1*npr.randn(num_params_to_fit)
    if pyap_options["model_number"]==3:
        x0 += nplog(15)-6*nplog(10)  # have to rescale because of Cm, this line is NOT general currently
    sigma0 = 0.1
    print "x0:", x0
    print "sigma0:", sigma0
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        #for q in X:
        #    print X
        es.tell(X, [sum_of_squares(x, ap_model, expt_trace) for x in X])
        es.disp()
    res = es.result()
    
    best_lnGs = res[0]
    best_sos = res[1]
    best_sigma = compute_initial_sigma(best_sos)
    best_ll = log_likelihood(best_sos, best_sigma)
    
    answer = np.concatenate((npexp(best_lnGs), [best_sigma, best_ll]))
    time_taken = time.time()-start
    
    with open(cmaes_log_file, "a") as outfile:
        outfile.write("{} iterations, {} min\n".format(res[4], round(time_taken/60.,1)))
    
    print "\n\nTime taken by this CMA-ES run: {} s\n\n".format(round(time_taken))
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
num_pts = len(expt_trace)



how_many_cmaes_runs = args.num_runs
cmaes_indices = range(how_many_cmaes_runs)

trace_start_time = time.time()
cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_log_likelihood_lnG(pyap_options["model_number"], expt_name, trace_name)
cmaes_log_file = cmaes_best_fits_file[:-3]+"log"
with open(cmaes_log_file, "w") as outfile:
    pass

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

trace_time_taken = time.time()-trace_start_time
print "\n\n{}, time taken: {} s = {} min\n\n".format(trace_name, round(trace_time_taken), round(trace_time_taken/60.))

# new AP model for plotting
ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(stimulus_magnitude, pyap_options["stimulus_duration_ms"], pyap_options["stimulus_period_ms"], pyap_options["stimulus_start_ms"])
ap_model.DefineModel(pyap_options["model_number"])
if (data_clamp_on < data_clamp_off):
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)        
ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])
ap_model.SetMembraneCapacitance(Cm)

best_ll_idx = np.argmax(best_boths[:, -1])
best_Gs = best_boths[best_ll_idx, :-2]
fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane voltage (mV)")
ax.grid()
ax.plot(expt_times, expt_trace, color='blue', label='Expt')
best_fit_trace = solve_for_voltage_trace(np.log(best_Gs), ap_model, expt_trace)
ax.plot(expt_times, best_fit_trace, color='red', label='Best ll')
max_V_idx = np.argmax(expt_trace)
ax.plot(expt_times[max_V_idx], expt_trace[max_V_idx], 'x', color="green", ms=9, mew=2, label='Expt max', zorder=100)
ax.legend(loc=1, fontsize=10)
fig.tight_layout()
fig.savefig(best_fit_png)
fig.savefig(best_fit_svg)
print best_fit_svg
print best_fit_png
plt.close()


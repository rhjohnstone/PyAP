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

# convert Cm and Istim to correct units, model-specific
if pyap_options["model_number"]==3:  # LR
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] * 1e-6
elif pyap_options["model_number"]==4:  # TT
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
elif pyap_options["model_number"]==5:  # OH
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
elif pyap_options["model_number"]==7:  # Pa
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-12
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-12

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


if data_clamp_on < data_clamp_off:
    solve_for_voltage_trace = solve_for_voltage_trace_with_initial_V
    print "Solving after setting V(0) = data(0)"
else:
    solve_for_voltage_trace = solve_for_voltage_trace_without_initial_V
    print "Solving without setting V(0) = data(0)"
    
    
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


cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_log_likelihood_lnG(pyap_options["model_number"], expt_name, trace_name)
cmaes_log_file = cmaes_best_fits_file[:-3]+"log"
best_boths = np.loadtxt(cmaes_best_fits_file)

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
fig, ax = plt.subplots(1, 1, figsize=(8,6))
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
#fig.savefig(best_fit_png)
#fig.savefig(best_fit_svg)
#print best_fit_svg
#print best_fit_png
plt.show()


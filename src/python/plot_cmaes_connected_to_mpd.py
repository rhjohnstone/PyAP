import matplotlib.pyplot as plt
import numpy as np
import pyap_setup as ps
import ap_simulator
import argparse
import sys


def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    temp_Gs = npexp(temp_lnG_params)
    solved = ap_model.SolveForVoltageTraceWithParams(temp_Gs)
    #if solved[-1] > -70:
    #    print temp_Gs
    return solved
    

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
args, unknown = parser.parse_known_args()

trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"

try:
    expt_times, expt_trace = np.loadtxt(trace_path, delimiter=',').T
except:
    sys.exit("Can't load expt trace: "+trace_path)

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

cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_log_likelihood_lnG(pyap_options["model_number"], expt_name, trace_name)
print "cmaes file:\n", cmaes_best_fits_file
try:
    cmaes_results = np.loadtxt(cmaes_best_fits_file)
    ndim = cmaes_results.ndim
    if ndim == 1:
        best_gs_sigma = cmaes_results[:-1]
    else:
        best_index = np.argmax(cmaes_results[:,-1])
        best_gs_sigma = cmaes_results[best_index,:-1]
    initial_gs = best_gs_sigma[:-1]
    initial_sigma = best_gs_sigma[-1]
except:
    sys.exit("Can't load CMA-ES")
    
theta_0 = np.concatenate((np.log(initial_gs), [initial_sigma]))
print "theta_0:", theta_0

mcmc_file, log_file, png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name)
try:
    sl_chain = np.loadtxt(mcmc_file)
    mpd_idx = np.argmax(sl_chain[:, -1])
    mpd_params = sl_chain[mpd_idx, :-1]
except:
    sys.exit("Can't load MCMC")
    
print "MPD params:", mpd_params

diff_vector = mpd_params - theta_0

num_x_pts = 121
diff = np.linspace(-0.1, 1.1, num_x_pts)

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

fig, ax = plt.subplots(1, 1, figsize=(6,4))
for d in diff:
    temp_params = theta_0 + d*diff_vector
    temp_trace = solve_for_voltage_trace_with_initial_V(temp_params, ap_model, expt_trace)
    ax.plot(expt_times, temp_trace, alpha=0.1)
fig.tight_layout()
plt.show(block=True)


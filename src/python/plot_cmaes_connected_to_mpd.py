import matplotlib.pyplot as plt
import numpy as np
import pyap_setup as ps
import ap_simulator
import argparse
import sys

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
args, unknown = parser.parse_known_args()

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
    print "initial_gs from cmaes:\n", initial_gs
except:
    sys.exit("Can't load CMA-ES")
    
theta_0 = np.concatenate((np.log(initial_gs), [initial_sigma]))
print theta_0


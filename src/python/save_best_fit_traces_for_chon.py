import ap_simulator
import numpy as np
import argparse
import pyap_setup as ps
import sys
import os

"""
Need to load pre-existing best-fit params, solve for the traces, then save these traces.
Copy these traces to chon-paper.
Write a script which loads and plots these and the original data.
"""


def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    try:
        return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    except ap_simulator.CPPException, e:
        print e.GetShortMessage
        sys.exit()


def where_to_save_best_fit(trace_name):
    best_fit_dir = os.path.expandvars("$DATA/PyAP_output/best_fits_for_chon/")
    if not os.path.exists(best_fit_dir):
        os.makedirs(best_fit_dir)
    return best_fit_dir + "{}_best_fit.txt".format(trace_name)


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
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

cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(pyap_options["model_number"], expt_name, trace_name)
best_boths = np.loadtxt(cmaes_best_fits_file)
best_fit_index = np.argmin(best_boths[:,-1])
best_params = best_boths[best_fit_index,:-1]
best_f = best_boths[best_fit_index,-1]

expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T

solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]
stimulus_magnitude = 0.
stimulus_duration = 1.
stimulus_start_time = 0.
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_params = len(original_gs)


ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
ap_model.DefineSolveTimes(solve_start, solve_end, solve_timestep)
ap_model.DefineModel(pyap_options["model_number"])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])
ap_model.UseDataClamp(pyap_options["data_clamp_on"], pyap_options["data_clamp_off"])
ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)

best_fit_trace = solve_for_voltage_trace(best_params, ap_model)
np.savetxt(where_to_save_best_fit(trace_name), best_fit_trace)


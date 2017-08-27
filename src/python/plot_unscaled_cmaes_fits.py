import pyap_setup as ps
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import ap_simulator


def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    try:
        return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    except ap_simulator.CPPException, e:
        print e.GetShortMessage
        print "temp_g_params:\n", temp_g_params
        print "original_gs:\n", original_gs
        sys.exit()


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"

expt_times, expt_trace = np.loadtxt(trace_path, delimiter=',').T

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
num_gs = len(g_parameters)

ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(0, 1, 1000, 0)  # no injected stimulus current
ap_model.DefineModel(pyap_options["model_number"])
ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])

cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(pyap_options["model_number"], expt_name, trace_name, args.unscaled)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
best_fits = np.loadtxt(cmaes_best_fits_file)
best_index = np.argmin(best_fits[:, -1])
best_params = best_fits[best_index, :-1]
best_trace = solve_for_voltage_trace(best_params, ap_model)
ax1.plot(expt_times, expt_trace, color='red', label='Expt')
ax1.plot(expt_times, best_trace, color='blue', label='Best fit')
plt.show(block=True)


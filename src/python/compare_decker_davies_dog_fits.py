#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ap_simulator
import numpy as np
import time
import pyap_setup as ps
import sys


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
        

npexp = np.exp
nplog = np.log
npinf = np.inf
npsum = np.sum
npsqrt = np.sqrt

sigma_uniform_lower = 1e-3
sigma_uniform_upper = 25.
omega = 0.5*nplog(10)  # s.d. of Normal priors on lnGs
two_omega_sq = 2.*omega**2
        
data_files = ["projects/PyAP/python/input/dog_teun_decker/traces/dog_AP_trace_150.csv", 
              "projects/PyAP/python/input/dog_teun_decker/traces/dog_AP_trace_151.csv",
              "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_150.csv", 
              "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_151.csv",]

trace_path = data_files[0]
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




if data_clamp_on < data_clamp_off:
    solve_for_voltage_trace = solve_for_voltage_trace_with_initial_V
    print "Solving after setting V(0) = data(0)"
else:
    solve_for_voltage_trace = solve_for_voltage_trace_without_initial_V
    print "Solving without setting V(0) = data(0)"


phi = 1.61803398875

all_time_start = time.time()


expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
num_pts = len(expt_trace)

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)
if pyap_options["model_number"] == 1:
    solve_end = 100  # just for HH
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)
num_params = num_gs + 1
log_gs = nplog(original_gs)

cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files_lnG(pyap_options["model_number"], expt_name, trace_name)
cmaes_log_file = cmaes_best_fits_file[:-3]+"log"

cmaes_output = np.loadtxt(cmaes_best_fits_file)
best_idx = np.argmin(cmaes_output[:, -1])
best_params = cmaes_output[best_idx, :-1]




best_AP = solve_for_voltage_trace(nplog(best_params[:-1]), ap_model, expt_trace)
cs = ['#1b9e77','#d95f02','#7570b3']

ax_y = 3
lw = 1
fig = plt.figure(figsize=(phi*ax_y,ax_y))
decker1 = fig.add_subplot(221)
decker2 = fig.add_subplot(223, sharex=decker1, sharey=decker1)
davies1 = fig.add_subplot(222, sharex=decker1)
davies2 = fig.add_subplot(224, sharex=decker1, sharey=davies1)
decker1.grid()
decker1.set_title(model_name)
decker1.set_xlabel('Time (ms)')
decker1.set_ylabel('Membrane voltage (mV)')
decker1.plot(expt_times, expt_trace, label=trace_name, lw=lw, color=cs[1])
decker1.plot(expt_times, best_AP, label="MPD", lw=lw, color=cs[0])
decker1.plot(expt_times, best_AP + 2*best_params[-1], label=r"MPD $\pm 2\sigma$", lw=lw, color=cs[2], ls="--")
decker1.plot(expt_times, best_AP - 2*best_params[-1], lw=lw, color=cs[2], ls="--")
decker1.legend(fontsize=10)
fig.tight_layout()
#fig.savefig(best_fit_png)
plt.show()




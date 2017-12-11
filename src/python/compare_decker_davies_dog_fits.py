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
              "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_150.csv", 
              "projects/PyAP/python/input/dog_teun_decker/traces/dog_AP_trace_151.csv",
              "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_151.csv",]

best_APs = []
model_names = []
best_sigmas = []
expt_traces = []
trace_numbers = []
MPDs = []
for trace_path in data_files:
    #trace_path = data_files[0]
    split_trace_path = trace_path.split('/')
    expt_name = split_trace_path[4]
    trace_name = split_trace_path[-1][:-4]
    options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"
    
    trace_numbers.append(int(trace_name.split("_")[-1]))

    pyap_options = {}
    with open(options_file, 'r') as infile:
        for line in infile:
            (key, val) = line.split()
            if (key == "model_number") or (key == "num_solves"):
                val = int(val)
            else:
                val = float(val)
            pyap_options[key] = val

    expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
    num_pts = len(expt_trace)
    
    expt_traces.append(np.copy(expt_trace))
            
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




    protocol = 1
    solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)
    if pyap_options["model_number"] == 1:
        solve_end = 100  # just for HH
    original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
    
    model_names.append(model_name)
    
    num_gs = len(original_gs)
    num_params = num_gs + 1
    log_gs = nplog(original_gs)

    cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files_lnG(pyap_options["model_number"], expt_name, trace_name)
    cmaes_log_file = cmaes_best_fits_file[:-3]+"log"

    cmaes_output = np.loadtxt(cmaes_best_fits_file)
    try:
        best_idx = np.argmin(cmaes_output[:, -1])
        best_params = cmaes_output[best_idx, :-1]
        best_mpd = cmaes_output[best_idx, -1]
    except:  # remove once Davies CMA-ES has finished (again)
        best_params = np.concatenate((original_gs,[0.5]))
        best_mpd = 0
    best_sigmas.append(best_params[-1])
    MPDs.append(-best_mpd)

    best_AP = solve_for_voltage_trace(nplog(best_params[:-1]), ap_model, expt_trace)
    best_APs.append(np.copy(best_AP))

cs = ['#1b9e77','#d95f02','#7570b3']

ax_y = 8
lw = 1
fig, axs = plt.subplots(2, 2, figsize=(phi*ax_y,ax_y), sharex=True, sharey=True)
for j in xrange(2):
    axs[j,0].set_ylabel('Membrane voltage (mV)')
    axs[1,j].set_xlabel('Time (ms)')

for i in xrange(2):
    axs[0,i].set_title(model_names[i])
    for j in xrange(2):
        axs[i,j].grid()
        idx = 2*i + j
        axs[i,j].plot(expt_times, expt_traces[idx], label="AP {}".format(trace_numbers[idx]), lw=lw, color=cs[1])
        axs[i,j].plot(expt_times, best_APs[idx], label=r"\log(MPD) \propto_+ {}$".format(round(MPDs[idx],1)), lw=lw, color=cs[0])
        axs[i,j].plot(expt_times, best_APs[idx] + 2*best_sigmas[idx], label=r"$MPD \pm 2\sigma$", lw=lw, color=cs[2], ls="--")
        axs[i,j].plot(expt_times, best_APs[idx] - 2*best_sigmas[idx], lw=lw, color=cs[2], ls="--")
        axs[i,j].legend(fontsize=12)
fig.tight_layout()
print best_fit_png
fig.savefig(best_fit_png)
plt.show()




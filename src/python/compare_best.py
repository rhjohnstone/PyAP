import ap_simulator  
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp
import argparse
import matplotlib.pyplot as plt


def solve_for_voltage_trace(temp_g_params):
    ap_model.SetToModelInitialConditions()
    return ap_model.SolveForVoltageTraceWithParams(temp_g_params)


def log_likelihood(params):
    temp_g_params, temp_sigma = params[:-1], params[-1]
    temp_trace = solve_for_voltage_trace(temp_g_params)
    return -len(temp_trace)*np.log(temp_sigma) - np.sum((temp_trace-expt_trace)**2)/(2*temp_sigma**2)


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
#parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
#parser.add_argument("--non-adaptive", action="store_true", help="do not adapt proposal covariance matrix", default=False)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"
expt_params_file = '/'.join( split_trace_path[:5] ) + "/expt_params.txt"

trace_number = int(trace_name.split('_')[-1])

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

"""mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name, args.unscaled, args.non_adaptive)

chain = np.loadtxt(mcmc_file)
best_idx = np.argmax(chain[:,-1])
print "Best index:", best_idx
best_params = chain[best_idx, :-1]"""

best_params = np.array([  9.93699070e+01,   8.71228491e-05,   2.56609157e-03,   1.50567124e-01,
   5.33947762e-02,   1.42294579e-03,   1.05740016e-03,   2.59690047e+01,
   2.08352354e-02,   1.28487507e-08,   7.70465562e-10,   1.05590032e-03,
   6.17794919e-03,   2.47998237e-01])
print "best:", best_params

true_params = np.concatenate((np.loadtxt(expt_params_file)[trace_number], [0.25]))
print "true:", true_params

diff = true_params - best_params

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_params = len(original_gs)+1  # include sigma

num_pts = 1001
lls = np.zeros(num_pts)
points = np.zeros((num_pts, num_params))
for j in xrange(num_params):
    points[:, j] = np.linspace(best_params[j] - 0.5*diff[j], best_params[j] + 1.5*diff[j], num_pts)
    
x_range = np.linspace(-0.5, 1.5, num_pts)

best_index = np.where(points==best_params)[0][0]
true_index = np.where(points==true_params)[0][0]

print best_index, true_index

try:
    expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
except:
    sys.exit( "\n\nCan't find (or load) {}\n\n".format(trace_path) )

solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]


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

fig, axs = plt.subplots(2, 2, sharex=True)#, sharey=True)
axs = axs.flatten()
for j, a in enumerate(range(4,8)):
    axs[j].grid()
    axs[j].set_ylabel("log-likelihood")
    axs[j].set_title(r"$10^{-"+str(a)+"}, 10^{-"+str(a+2)+"}$")
    ap_model.SetTolerances(10**-a, 10**-(a+2))
    for i in xrange(num_pts):
        lls[i] = log_likelihood(points[i, :])
    axs[j].plot(x_range, lls, color='blue', lw=1)
    axs[j].axvline(x_range[best_index], color='red', label='best', lw=2)
    axs[j].axvline(x_range[true_index], color='green', label='true', lw=2)
fig.tight_layout()
fig.savefig("synth_ohara_lls_tols.png")
plt.show(block=True)






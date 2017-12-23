import pyap_setup as ps
import ap_simulator
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import numpy.random as npr
from time import time
import sys

npexp = np.exp

def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    

seed = 3
npr.seed(seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
parser.add_argument("-T", "--num-samples", type=int, help="number of AP samples to plot", default=0)
parser.add_argument("-n", "--num-expts", type=int, help="number of traces to construct Gary-predictive from", required=True)
#parser.add_argument("-x", "--num-pts", type=int, help="number of x points to plot Gary-predictive for", required=True)
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

try:
    expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
except:
    sys.exit( "\n\nCan't find (or load) {}\n\n".format(trace_path) )


split_trace_name = trace_name.split("_")

        
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)

g_labels = ["${}$".format(g) for g in g_parameters]

m_true = np.log(original_gs)
sigma2_true = 0.01

N_e = args.num_expts

gary_predictives = []

for i in xrange(num_gs):
    garyfile, garypng = ps.gary_predictive_file(expt_name, N_e, i)
    gary_predictives.append(np.loadtxt(garyfile))

cs = ['#1b9e77','#d95f02','#7570b3']



"""fig, axs = plt.subplots(1, 2, figsize=(7,3), sharey=True)
axs[0].set_ylabel("Cumulative dist.")
p_s = [0, 11]
for i, p in enumerate(p_s):
    axs[i].grid()
    axs[i].set_xlabel(r"$\log({})$".format(g_parameters[p]), fontsize=16)
    axs[i].set_xlim(gary_predictives[p][0,0], gary_predictives[p][-1,0])
    axs[i].plot(*gary_predictives[p].T, lw=3, color=cs[0])
    gary_predictive_samples = np.interp(rand_samples, gary_predictives[p][:,1], gary_predictives[p][:,0])
    for t in xrange(T):
        axs[i].plot([gary_predictives[p][0,0], gary_predictive_samples[t]], [rand_samples[t], rand_samples[t]], color=cs[1], lw=2)
        axs[i].plot([gary_predictive_samples[t], gary_predictive_samples[t]], [0, rand_samples[t]], color=cs[2], lw=2)
fig.tight_layout()
plt.show()"""



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


T = args.num_samples
rand_samples = npr.rand(T)
fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.grid()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane voltage (mV)")
start = time()
for t in xrange(T):
    temp_lnGs = [np.interp(rand_samples[t], gary_predictives[p][:,1], gary_predictives[p][:,0]) for p in xrange(num_gs)]
    ax.plot(expt_times, solve_for_voltage_trace_with_initial_V(temp_lnGs, ap_model, expt_trace), alpha=0.01, color='black')
time_taken = time()-start
print "Time taken for {} solves and plots: {} s = {} min".format(T, int(time_taken), round(time_taken/60., 1))
ax.plot([], [], label="{} samples".format(T), color='black')
ax.legend(loc=1)
fig.tight_layout()
plt.show()



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
import pandas as pd
import seaborn as sns

npexp = np.exp

def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))


def solve_for_voltage_trace_with_block(temp_lnG_params, ap_model, expt_trace, dose):
    ap_model.SetToModelInitialConditions()
    temp_Gs = npexp(temp_lnG_params)
    temp_Gs = apply_moxi_blocks(temp_Gs, dose)
    #ap_model.SetVoltage(expt_trace[0])
    return ap_model.SolveForVoltageTraceWithParams(temp_Gs)


def fraction_block(dose,hill,IC50):
    return 1. - 1./(1.+(1.*dose/IC50)**hill)
    
def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)
    
def ic50_to_pic50(ic50): # IC50 in uM
    return 6-np.log10(ic50)
    

                        
def apply_moxi_blocks(temp_G_params, dose):
    hill = 1
    for i, p in enumerate(block_indices):
        temp_G_params[p] *= (1.-fraction_block(dose, hill, ic50s[i]))
    return temp_G_params
    

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

g_parameters = ['G_{Na}', 'G_{CaL}', 'G_{K1}', 'G_{pK}',
                        'G_{Ks}', 'G_{Kr}', 'G_{pCa}', 'G_{bCa}',
                        'k_{NaCa}', 'P_{NaK}', 'G_{to1}', 'G_{to2}',
                        'G_{bCl}', 'G_{NaL}']

if pyap_options["model_number"]==6:
    trace_number = int(split_trace_name[-1])
    block_indices = [0, 1, 4, 5]
    ic50s = [206.7, 158., 158., 29.]
elif pyap_options["model_number"]==4:
    trace_number = int(split_trace_name[-2])
        
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


#fig, axs = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
#for j in xrange(2):
#    axs[j].set_xlabel("Time (ms)")
#    axs[j].grid()
#axs[0].set_ylabel("Membrane voltage (mV)")

#axs[0].set_title("Experimental")
#axs[1].set_title("Predicted")

control_apd90s = np.zeros(T)
unif_samples = npr.rand(T, num_gs)
start = time()
for t in xrange(T):
    temp_lnGs = [np.interp(unif_samples[t,p], gary_predictives[p][:,1], gary_predictives[p][:,0]) for p in xrange(num_gs)]
    test_trace = solve_for_voltage_trace_with_initial_V(temp_lnGs, ap_model, expt_trace)
    #axs[1].plot(expt_times, test_trace, alpha=0.01, color='blue')
    control_apd90s[t] = ps.compute_apd90(expt_times, test_trace, data_clamp_on)
time_taken = time()-start
print "Time taken for {} solves and plots: {} s = {} min".format(T, int(time_taken), round(time_taken/60., 1))
#axs[1].plot([], [], label="Control", color='blue')
#fig.tight_layout()
#fig_png = "{}_trace_{}_{}_samples.png".format(expt_name, trace_number, T)
#print fig_png
#fig.savefig(fig_png)
#plt.show()

expt_apd90s_control = np.zeros(N_e)
for i in xrange(N_e):
    plot_trace_number = 150 + i
    plot_trace_path = "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_{}.csv".format(plot_trace_number)

    expt_times_for_plotting, expt_trace_for_plotting = np.loadtxt(plot_trace_path, delimiter=',').T
    #axs[0].plot(expt_times_for_plotting, expt_trace_for_plotting, color='blue')
    print "Trace", plot_trace_number
    expt_apd90s_control[i] = ps.compute_apd90(expt_times_for_plotting, expt_trace_for_plotting, data_clamp_on)

expt_apd90s_moxi = np.zeros(N_e)
for i in xrange(N_e):
    plot_trace_number = 400 + i
    plot_trace_path = "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_{}.csv".format(plot_trace_number)

    expt_times_for_plotting, expt_trace_for_plotting = 1000.*np.loadtxt(plot_trace_path, delimiter=',').T
    #axs[0].plot(expt_times_for_plotting, expt_trace_for_plotting, color='red')
    print "Trace", plot_trace_number
    expt_apd90s_moxi[i] = ps.compute_apd90(expt_times_for_plotting, expt_trace_for_plotting, data_clamp_on)

expts = ["Control"]*N_e + ["K$^+$, Moxi."]*N_e
colours = ["blue"]*N_e + ["red"]*N_e
expts_apds = np.concatenate((expt_apd90s_control, expt_apd90s_moxi))

d = {"Experiment": expts, "colours": colours, "$APD_{90}s$": expts_apds}

expt_df = pd.DataFrame(data=d)

ax = sns.swarmplot(x="Experiment", y="apds", data=expt_df, hue="colours")
plt.show()

#axs[0].plot([], [], color='blue', label='Control')
#axs[0].plot([], [], color='red', label="K$^+$, Moxi.")

#axs[0].legend(loc=1)

moxi_conc = 10
new_extra_K_conc = 4
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
ap_model.SetExtracellularPotassiumConc(new_extra_K_conc)
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])


unif_samples = npr.rand(T, num_gs)
moxi_apd90s = np.zeros(T)
start = time()
for t in xrange(T):
    temp_lnGs = [np.interp(unif_samples[t,p], gary_predictives[p][:,1], gary_predictives[p][:,0]) for p in xrange(num_gs)]
    test_trace = solve_for_voltage_trace_with_block(temp_lnGs, ap_model, expt_trace, moxi_conc)
    #axs[1].plot(expt_times, test_trace, alpha=0.01, color='red')
    moxi_apd90s[t] = ps.compute_apd90(expt_times, test_trace, data_clamp_on)
time_taken = time()-start
print "Time taken for {} solves and plots: {} s = {} min".format(T, int(time_taken), round(time_taken/60., 1))
#axs[1].plot([], [], label="K$^+$, Moxi.", color='red')
#axs[1].legend(loc=1)
#fig.tight_layout()
#fig_png = "{}_trace_{}_{}_samples_control_and_moxi_predictions.png".format(expt_name, trace_number, T)
#print fig_png
#fig.savefig(fig_png)

apd90_fig = plt.figure(figsize=(4,3))
apd90_ax = apd90_fig.add_subplot(111)
apd90_ax.grid()
apd90_ax.hist(control_apd90s, bins=30, normed=True, color='blue', alpha=0.5, lw=0)
apd90_ax.hist(moxi_apd90s, bins=30, normed=True, color='red', alpha=0.5, lw=0)
for n in xrange(N_e):
    apd90_ax.axvline(expt_apd90s_control[n], color='orange')
    apd90_ax.axvline(expt_apd90s_moxi[n], color='green')
apd90_ax.set_xlabel("APD90 (ms)")
apd90_ax.set_ylabel("Normalised frequency")
apd90_fig.tight_layout()
apd90_fig_png = "{}_trace_{}_{}_samples_control_and_moxi_apd90_predictions.png".format(expt_name, trace_number, T)
print apd90_fig_png
#apd90_fig.savefig(apd90_fig_png)

plt.show(block=True)


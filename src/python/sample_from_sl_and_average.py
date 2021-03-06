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
    temp_Gs = npexp(temp_lnG_params)
    solved = ap_model.SolveForVoltageTraceWithParams(temp_Gs)
    #if solved[-1] > -70:
    #    print temp_Gs
    return solved


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
    

#seed = 3
#npr.seed(seed)

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



try:
    expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
except:
    sys.exit( "\n\nCan't find (or load) {}\n\n".format(trace_path) )


split_trace_name = trace_name.split("_")


if pyap_options["model_number"]==6:
    trace_number = int(split_trace_name[-1])
    block_indices = [0, 1, 4, 5]
    ic50s = [206.7, 158., 158., 29.]
elif pyap_options["model_number"]==4 or pyap_options["model_number"]==7:
    trace_number = int(split_trace_name[-2])
        
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)

g_labels = ["${}$".format(g) for g in g_parameters]

m_true = np.log(original_gs)
sigma2_true = 0.01

N_e = args.num_expts

sl_chains = []
chain_lengths = []
for n in xrange(N_e):
    if expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
        temp_trace_name = "_".join(split_trace_name[:-2]) + "_{}_".format(100+n) + split_trace_name[-1]
    print "Trace:", temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    try:
        temp_chain = np.loadtxt(sl_mcmc_file, usecols=range(num_gs))
    except:
        print "Can't load", sl_mcmc_file
        continue
    saved_its = temp_chain.shape[0]
    if expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
        temp_chain = temp_chain[(3*saved_its)/5:, :]
    sl_chains.append(temp_chain)
    chain_lengths.append(temp_chain.shape[0])

length = chain_lengths[0]

num_chains = len(sl_chains)



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


T = args.num_samples



fig, axs = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
for j in xrange(2):
    axs[j].set_xlabel("Time (ms)")
    axs[j].grid()
axs[0].set_ylabel("Membrane voltage (mV)")

axs[0].set_title("Experimental")
axs[1].set_title("Predicted")

start = time()
rand_idx = npr.randint(0, length, size=(T, num_chains))
for t in xrange(T):
    lnG_samples = np.zeros((num_chains, num_gs))
    for n in xrange(num_chains):
        lnG_samples[n, :] = sl_chains[n][rand_idx[t, n]]
    print temp_lnGs
    temp_lnGs = lnG_samples.mean(axis=0)
    axs[1].plot(expt_times, solve_for_voltage_trace_with_initial_V(temp_lnGs, ap_model, expt_trace), alpha=0.01, color='blue')
time_taken = time()-start
print "Time taken for {} solves and plots: {} s = {} min".format(T, int(time_taken), round(time_taken/60., 1))
#axs[1].plot([], [], label="Control", color='blue')
#fig.tight_layout()
#fig_png = "{}_trace_{}_{}_samples.png".format(expt_name, trace_number, T)
#print fig_png
#fig.savefig(fig_png)
#plt.show()


for i in xrange(N_e):
    plot_trace_number = 100 + i
    if expt_name=="roche_ten_tusscher_correct_units":
        plot_trace_path = "projects/PyAP/python/input/roche_ten_tusscher/traces/Trace_2_2_{}_1.csv".format(plot_trace_number)
    elif expt_name=="roche_paci_correct_units":
        plot_trace_path = "projects/PyAP/python/input/roche_paci/traces/Trace_2_2_{}_1.csv".format(plot_trace_number)

    expt_times_for_plotting, expt_trace_for_plotting = np.loadtxt(plot_trace_path, delimiter=',').T
    axs[0].plot(expt_times_for_plotting, expt_trace_for_plotting, color='blue')


"""for i in xrange(N_e):
    plot_trace_number = 400 + i
    plot_trace_path = "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_{}.csv".format(plot_trace_number)

    expt_times_for_plotting, expt_trace_for_plotting = 1000.*np.loadtxt(plot_trace_path, delimiter=',').T
    axs[0].plot(expt_times_for_plotting, expt_trace_for_plotting, color='red')"""

#axs[0].plot([], [], color='blue', label='Control')
#axs[0].plot([], [], color='red', label="K$^+$, Moxi.")

#axs[0].legend(loc=1)

"""
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

rand_samples = npr.rand(T)


start = time()
for t in xrange(T):
    temp_lnGs = [np.interp(rand_samples[t], gary_predictives[p][:,1], gary_predictives[p][:,0]) for p in xrange(num_gs)]
    axs[1].plot(expt_times, solve_for_voltage_trace_with_block(temp_lnGs, ap_model, expt_trace, moxi_conc), alpha=0.01, color='red')
time_taken = time()-start
print "Time taken for {} solves and plots: {} s = {} min".format(T, int(time_taken), round(time_taken/60., 1))
axs[1].plot([], [], label="K$^+$, Moxi.", color='red')
axs[1].legend(loc=1)"""



fig.tight_layout()
fig_png = "{}_trace_{}_{}_sample_from_sl_and_then_mean.png".format(expt_name, trace_number, T)
print fig_png
fig.savefig(fig_png)
plt.show()

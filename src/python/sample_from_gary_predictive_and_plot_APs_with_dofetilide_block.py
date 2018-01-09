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
parser.add_argument("-T", "--num-samples", type=int, help="number of AP and block (together) samples", default=0)
parser.add_argument("-n", "--num-expts", type=int, help="number of traces to construct Gary-predictive from", required=True)
parser.add_argument("-m", "--model", type=int, help="dose-response model number", required=True)
parser.add_argument("-d", "--dose", type=float, help="Dofetilide concentration", required=True)
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
    indices_to_keep = [0, 1, 2, 3, 4, 6]  # which currents we are fitting
    #                  G_Na, G_CaL, G_K1, G_Kr, G_Ks, G_to
    indices_to_block = [0, 1, 2, 3, 4, 6]  # same order as channels, no G_NaL
elif pyap_options["model_number"]==5:  # OH
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
elif pyap_options["model_number"]==7:  # Pa
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-12
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-12
    indices_to_keep = [0, 1, 2, 3, 4, 9, 11]  # which currents we are fitting
    #                  G_Na, G_CaL, G_K1, G_Ks, G_Kr, G_to, G_f
    indices_to_block = [0, 1, 2, 4, 3, 9]  # same order as channels, no G_NaL
    
num_params_to_fit = len(indices_to_keep) + 1  # +1 for sigma

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(indices_to_keep)
log_gs = np.log(original_gs[indices_to_keep])

cmaes_final_state_vars_file = ps.cmaes_final_state_vars_file(pyap_options["model_number"], expt_name+"_2_paces", trace_name)
print "cmaes_final_state_vars_file:\n", cmaes_final_state_vars_file
cmaes_final_state_vars = np.loadtxt(cmaes_final_state_vars_file)
print "cmaes_final_state_vars:\n", cmaes_final_state_vars



def solve_for_voltage_trace_with_ICs(temp_G_params, ap_model, expt_trace):
    ap_model.SetStateVariables(cmaes_final_state_vars)
    #ap_model.SetVoltage(expt_trace[0])
    
    #temp_Gs[indices_to_keep] = temp_G_params
    try:
        return ap_model.SolveForVoltageTraceWithParams(temp_G_params)
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)



if data_clamp_on < data_clamp_off:
    solve_for_voltage_trace = solve_for_voltage_trace_with_ICs
    print "Solving after setting ICs to the cmaes ones"
else:
    sys.exit("This is just for Roche, so there should be some data-clamp.")


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

drug = "Dofetilide"
channels = ["Nav1.5-peak", "Cav1.2",  "Kir2.1", "hERG", "KvLQT1_mink",  "Kv4.3"]#,   "Nav1.5-late"]
#            G_Na             G_CaL,   G_K1,      G_Kr,     G_Ks,          G_to,         G_NaL
models = [1, 2]

num_channels = len(channels)

model = args.model
print "model", model
block_chains = []
for channel in channels:
    chain_file = "/data/coml-cardiac/hert3352/Dofetilide/{channel}/model_{model}/temperature_1/chain/Dofetilide_{channel}_model_{model}_temp_1_chain_nonhierarchical.txt".format(channel=channel, model=model)
    block_chain = np.loadtxt(chain_file, usecols=range(2))
    saved_its = block_chain.shape[0]
    block_chain = block_chain[saved_its/4:, :]
    if model==1:
        block_chain[:, 1] = 1.
    block_chains.append(block_chain)
block_length = block_chains[0].shape[0]

dose = args.dose

#unif_samples = npr.rand(T, num_gs)


fig, ax = plt.subplots(1, 1, figsize=(5,4))

ax.set_xlabel("Time (ms)")
ax.grid()
ax.set_ylabel("Membrane voltage (mV)")

#ax.set_title(r"Predicted {} $\mu$M {}, Model {}".format(dose, drug, model))


start = time()

for t in xrange(T):
    temp_Gs = np.copy(original_gs)
    unif_samples = npr.rand(num_gs)
    temp_lnG_samples = [np.interp(unif_samples[p], gary_predictives[p][:,1], gary_predictives[p][:,0]) for p in xrange(num_gs)]
    temp_G_samples = npexp(temp_lnG_samples)
    temp_Gs[indices_to_keep] = temp_G_samples
    ax.plot(expt_times, solve_for_voltage_trace_with_ICs(temp_Gs, ap_model, expt_trace), alpha=0.2, color='blue')



for t in xrange(T):
    temp_Gs = np.copy(original_gs)
    unif_samples = npr.rand(num_gs)
    temp_lnG_samples = [np.interp(unif_samples[p], gary_predictives[p][:,1], gary_predictives[p][:,0]) for p in xrange(num_gs)]
    temp_G_samples = npexp(temp_lnG_samples)
    temp_Gs[indices_to_keep] = temp_G_samples
    block_idx = npr.randint(0, block_length, num_channels)
    blocks = np.zeros(num_channels)
    for c in xrange(num_channels):
        pic50, hill = block_chains[c][block_idx[c], :]
        blocks[c] = fraction_block(dose, hill, pic50_to_ic50(pic50))
    print blocks
    temp_Gs[indices_to_block] *= (1.-blocks)
    #print temp_Gs
    ax.plot(expt_times, solve_for_voltage_trace_with_ICs(temp_Gs, ap_model, expt_trace), alpha=0.2, color='red')
time_taken = time()-start
print "Time taken for {} solves and plots: {} s = {} min".format(T, int(time_taken), round(time_taken/60., 1))
#axs[1].plot([], [], label="Control", color='blue')
#fig.tight_layout()
#fig_png = "{}_trace_{}_{}_samples.png".format(expt_name, trace_number, T)
#print fig_png
#fig.savefig(fig_png)
#plt.show(block=True)


"""for i in xrange(N_e):
    plot_trace_number = 100 + i
    if expt_name=="roche_ten_tusscher_correct_units_subset":
        plot_trace_path = "projects/PyAP/python/input/roche_ten_tusscher_correct_units_subset/traces/Trace_2_2_{}_1.csv".format(plot_trace_number)
    elif expt_name=="roche_paci_correct_units_subset":
        plot_trace_path = "projects/PyAP/python/input/roche_paci_correct_units_subset/traces/Trace_2_2_{}_1.csv".format(plot_trace_number)

    expt_times_for_plotting, expt_trace_for_plotting = np.loadtxt(plot_trace_path, delimiter=',').T
    axs[0].plot(expt_times_for_plotting, expt_trace_for_plotting, color='blue')"""


fig.tight_layout()
fig_png = "{}_trace_{}_{}_samples_with_{}_uM_dofetilide_block_model_{}.png".format(expt_name, trace_number, T, dose, model)
print fig_png
fig.savefig(fig_png)
plt.show()

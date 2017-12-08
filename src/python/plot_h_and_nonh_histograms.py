import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import matplotlib.pyplot as plt
import itertools as it


#python_seed = 1
#npr.seed(python_seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("-n", "--num-traces", type=int, help="which hMCMC to use", required=True)

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
expt_params_file = '/'.join( split_trace_path[:5] ) + "/expt_params.txt"

split_trace_name = trace_name.split("_")
first_trace_number = int(split_trace_name[-1])

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
num_gs = len(original_gs)

g_labels = ["${}$".format(g) for g in g_parameters]
    
N_e = args.num_traces

expt_params = np.loadtxt(expt_params_file)[:N_e,:]

parallel = True

color_idx = np.linspace(0, 1, N_e)
mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_lnG_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
h_chain = np.loadtxt(mcmc_file)
saved_its, d = h_chain.shape
for i in xrange(num_gs):
    fig, (ax2,ax) = plt.subplots(1,2,figsize=(8,4), sharex=True,sharey=True)
    ax.grid()
    ax2.grid()
    ax2.set_xlabel(g_labels[i])
    ax.set_xlabel(g_labels[i])
    ax2.set_ylabel("Normalised frequency")
    ax2.set_title('Single-level')
    ax.set_title('Hierarchical')
    for n in xrange(N_e):
        idx = (2+n)*num_gs + i
        colour = plt.cm.winter(color_idx[n])
        temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(n)
        sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
        single_chain = np.loadtxt(sl_mcmc_file, usecols=[i])
        
        ax2.hist(single_chain, normed=True, bins=40, color=colour, edgecolor=colour, alpha=4./N_e)
    

        ax.hist(h_chain[:, idx], normed=True, bins=40, color=colour, edgecolor=colour, alpha=4./N_e)
        
    
        line = ax.scatter(expt_params[n, i], 0, marker='x', c='red', zorder=10)
        line.set_clip_on(False)
        
        line = ax2.scatter(expt_params[n, i], 0, marker='x', c='red', zorder=10)
        line.set_clip_on(False)
    
    ax2.set_ylim(0, ax2.get_ylim()[1])
    for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    fig.tight_layout()
    #fig.savefig(hpng_dir+"{}_h_and_nonh_{}_traces_{}.png".format(expt_name, N_e, g_parameters[i]))

    plt.show(block=True)


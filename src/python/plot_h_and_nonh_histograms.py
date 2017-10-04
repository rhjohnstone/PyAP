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
requiredNamed.add_argument("-s", "--series", action="store_true", help="plot parallel hMCMC", default=False)

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

parallel = not args.series
print parallel


"""for n, i in it.product(range(2), range(num_gs)):
    print n, i
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    if n == 0:
        xlabel = r"\hat{" + g_parameters[i] +"}$"
        figname = "trace_{}_top_{}.png".format(n, g_parameters[i])
    elif n == 1:
        xlabel = r"\sigma_{" + g_parameters[i] +"}^2$"  # need to check if this squared is correct
        figname = "trace_{}_sigma_{}_squared.png".format(n, g_parameters[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised frequency")
    idx = (2+n)*num_gs + i
    ax.hist(chain[burn:, idx], bins=40, color='blue', edgecolor='blue')"""
    #fig.savefig(png_dir+figname)
    #plt.close()

    
N_e = args.num_traces

expt_params = np.loadtxt(expt_params_file)[:N_e,:]

color_idx = np.linspace(0, 1, N_e)
mcmc_file, log_file, hpng_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
chain = np.loadtxt(mcmc_file)
saved_its, d = chain.shape
chain = chain[saved_its/2:, :]
T, d = chain.shape
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
        
        single_trace_name = trace_name[:-1]+str(n)
        mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, single_trace_name, unscaled=True, non_adaptive=False)
        single_chain = np.loadtxt(mcmc_file, usecols=[i])
        
        ax2.hist(single_chain[single_chain.shape[0]/2:], normed=True, bins=40, color=colour, edgecolor=colour, alpha=0.8)
    

        ax.hist(chain[:, idx], normed=True, bins=40, color=colour, edgecolor=colour, alpha=0.8)
        
    
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
    fig.savefig(hpng_dir+"{}_h_and_nonh_{}_traces_{}.png".format(expt_name, N_e, g_parameters[i]))

    plt.close()


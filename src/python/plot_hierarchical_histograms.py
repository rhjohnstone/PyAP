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

parallel = True


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
    
    
which_g = 0
    
N_e = args.num_traces
mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
chain = np.loadtxt(mcmc_file)
saved_its, d = chain.shape
chain = chain[saved_its/2:, :]
T, d = chain.shape
for i in xrange(num_gs):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_xlabel(g_labels[i])
    ax.set_ylabel("Normalised frequency")
    for n in xrange(N_e):
        idx = (2+n)*num_gs + i
        ax.hist(chain[:, idx], normed=True, bins=40, color='blue', edgecolor=None, alpha=0.8)
    plt.xticks(rotation=30)
    fig.tight_layout()

plt.show(block=True)


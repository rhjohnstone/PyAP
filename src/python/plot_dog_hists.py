import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import sys
import pyap_setup as ps

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
requiredNamed.add_argument("--num-traces", type=int, help="number of traces to plot hists of", required=True)
parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=True)
parser.add_argument("--non-adaptive", action="store_true", help="do not adapt proposal covariance matrix", default=False)
parser.add_argument("-b", "--burn", type=int, help="what fraction of samples to discard", default=4)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
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

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])

fig, axs = plt.subplots(5, 3)
axs = axs.flatten()
ids = range(15)
mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
color_idx = np.linspace(0, 1, args.num_traces)

for t in xrange(args.num_traces):
    trace_name = "dog_AP_trace_{}".format(150+t)
    mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name, True, False)
    chain = np.loadtxt(mcmc_file, usecols=range(15))
    saved_its, _ = chain.shape
    burn = saved_its/args.burn
    for i in ids:
        if i<ids[-1]:
            label = "${}$".format(g_parameters[i])
        else:
            label = r"$\sigma$"
        col = mymap(color_idx[t])
        axs[i].grid()
        axs[i].hist(chain[burn:, i], bins=40, color=col, edgecolor=col, alpha=0.2)
        axs[i].yaxis.set_major_locator(plt.NullLocator())
        axs[i].set_xlabel(label)
fig.tight_layout()
plt.show(block=True)


import pyap_setup as ps
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf    

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
parser.add_argument("--non-adaptive", action="store_true", help="do not adapt proposal covariance matrix", default=False)
parser.add_argument("--hierarchical", action="store_true", help="plot for hMCMC", default=False)
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

pyap_options = {}
with open(options_file, 'r') as infile:
    for line in infile:
        (key, val) = line.split()
        if (key == "model_number") or (key == "num_solves"):
            val = int(val)
        else:
            val = float(val)
        pyap_options[key] = val

mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name, args.unscaled, args.non_adaptive)

chain = np.loadtxt(mcmc_file)

saved_its, num_params = chain.shape
burn = saved_its/4

G_Na_samples = chain[burn:, 0]

fig = plt.figure()
ax = fig.add_subplot(111)
plot_acf(G_Na_samples, ax=ax, use_vlines=False)

plt.show(block=True)


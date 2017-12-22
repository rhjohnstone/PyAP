import pyap_setup as ps
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import numpy.random as npr
from time import time
import sys

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

T = args.num_samples
rand_samples = npr.rand(T)
fig, axs = plt.subplots(1,2,figsize=(8,3))
axs[0].set_ylabel("Cumulative dist.")
p_s = [0, 11]
for i, p in enumerate(p_s):
    axs[i].grid()
    axs[i].set_xlabel(r"$\log({})$".format(g_parameters[p]), fontsize=16)
    axs[i].set_xlim(gary_predictives[p][0,0], gary_predictives[p][-1,0])
    axs[i].plot(*gary_predictives[p].T, lw=2, color=cs[0])
    gary_predictive_samples = np.interp(rand_samples, gary_predictives[p][:,1], gary_predictives[p][:,0])
    for t in xrange(T):
        axs[i].plot([gary_predictives[p][0,0], gary_predictive_samples[t]], [rand_samples[t], rand_samples[t]], color=cs[1])
        axs[i].plot([gary_predictive_samples[t], gary_predictive_samples[t]], [0, rand_samples[t]], color=cs[2])
fig.tight_layout()
plt.show()


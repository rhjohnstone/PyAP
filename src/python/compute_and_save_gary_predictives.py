import pyap_setup as ps
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
import argparse
import numpy.random as npr
import os
from time import time

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
parser.add_argument("-T", "--num-samples", type=int, help="number of samples to plot prior predictive from", default=0)
parser.add_argument("-n", "--num-expts", type=int, help="number of traces to construct Gary-predictive from", required=True)
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


parallel = True


cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 101
nums_expts = [2, 4, 8, 16, 32]

labels = ("True",) + tuple(["$N_e = {}$".format(n) for n in nums_expts])
lines = ()

total_nums_expts = len(nums_expts)
color_idx = np.linspace(0, 1, total_nums_expts)

normpdf = norm.pdf
normrvs = norm.rvs
invgammarvs = invgamma.rvs

N_e = args.num_expts

xs = []
sl_chains = []
mins = 1e9*np.ones(num_gs)
print mins
maxs = -1e9*np.ones(num_gs)
for n in xrange(N_e):
    if (pyap_options["model_number"]==2) or (pyap_options["model_number"]==5):  # synth BR/OH
        temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(n)
    elif (pyap_options["model_number"]==2) or (pyap_options["model_number"]==6):  # Davies
        temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(150+n)
    elif pyap_options["model_number"]==4:  # TT
        temp_trace_name = "_".join(split_trace_name[:-2]) + "_{}_".format(100+n) + split_trace_name[-1]
    print "Trace:", temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    temp_chain = np.loadtxt(sl_mcmc_file, usecols=range(num_gs))
    sl_chains.append(temp_chain)
    temp_mins = np.min(temp_chain, axis=0)
    temp_maxs = np.max(temp_chain, axis=0)
    where_new_min = np.where(temp_mins < mins)
    where_new_max = np.where(temp_maxs > maxs)
    mins[where_new_min] = temp_mins[where_new_min]
    maxs[where_new_max] = temp_maxs[where_new_max]
    print mins

num_pts = 51
for i in xrange(num_gs):
    x = np.linspace(0.1*original_gs[i], 1.9*original_gs[i], num_pts)

sys.exit()
for i in xrange(num_gs):
    xs.append(np.linspace(m_true[i]-2*np.sqrt(sigma2_true), m_true[i]+2*np.sqrt(sigma2_true), num_pts))
    x = xs[i]
    for a, N_e in enumerate(nums_expts):
        # Gary approximation
        sl_chains = []
        for n in xrange(N_e):
            if (pyap_options["model_number"]==2) or (pyap_options["model_number"]==5):  # synth BR/OH
                temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(n)
            elif pyap_options["model_number"]==4:  # TT
                temp_trace_name = "_".join(split_trace_name[:-2]) + "_{}_".format(n) + split_trace_name[-1]
            print "Trace:", temp_trace_name
            sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
            temp_chain = np.loadtxt(sl_mcmc_file, usecols=[i])
            sl_chains.append(temp_chain)
        gary_pred = np.zeros(num_pts)
        for t in xrange(args.num_samples):
            samples = np.zeros(N_e)
            for n in xrange(N_e):
                rand_idx = npr.randint(0,len(sl_chains[n]))
                samples[n] = sl_chains[n][rand_idx]
            #if t%100==0:
            #    print "samples:", samples
            loc, scale = norm.fit(samples)
            gary_pred += norm.pdf(x, loc=loc, scale=scale)
        gary_pred /= args.num_samples
        
        # Posterior predictive
        hmcmc_file, log_file, h_png_dir, pdf_dir = ps.hierarchical_lnG_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
        h_chain = np.loadtxt(hmcmc_file,usecols=[i, num_gs+i])
        saved_its = h_chain.shape[0]
        start = time()
        
        post_pred = np.zeros(num_pts)
        if args.num_samples == 0:
            T = saved_its
            for t in xrange(T):
                post_pred += normpdf(x, loc=h_chain[t, 0], scale=np.sqrt(h_chain[t, 1]))
        else:
            T = args.num_samples
            rand_idx = npr.randint(0, saved_its, T)
            for t in xrange(T):
                post_pred += normpdf(x, loc=h_chain[rand_idx[t], 0], scale=np.sqrt(h_chain[rand_idx[t], 1]))
        post_pred /= T
        tt = time()-start
        line, = ax2.plot(x, post_pred, lw=2, color=colour, label="$N_e = {}$".format(N_e))
        if i==0:
            lines += (line,)
        ax1.plot(x, gary_pred, lw=2, color=colour, label="$N_e = {}$".format(N_e))
        

        


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
parser.add_argument("-T", "--num-samples", type=int, help="number of samples to plot prior predictive from", default=0)
parser.add_argument("-n", "--num-expts", type=int, help="number of traces to construct Gary-predictive from", required=True)
parser.add_argument("-x", "--num-pts", type=int, help="number of x points to plot Gary-predictive for", required=True)
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

normpdf = norm.pdf
normrvs = norm.rvs

N_e = args.num_expts

xs = []
sl_chains = []
mins = 1e9*np.ones(num_gs)
maxs = -1e9*np.ones(num_gs)
chain_lengths = np.zeros(N_e)
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
    chain_lengths[n] = temp_chain.shape[0]
    
length = chain_lengths[0]
assert(np.all(chain_lengths==length))

num_pts = args.num_pts
xs = np.zeros((num_gs, num_pts))
for i in xrange(num_gs):
    xs[i, :] = np.linspace(mins[i]-0.1, maxs[i]+0.1, num_pts)



T = args.num_samples
gary_predictives = np.zeros((num_gs, num_pts))
rand_idx = npr.randint(0, length, size=(N_e, T))  # don't know if this will cause memory/speed issues
for i in xrange(num_gs):
    for t in xrange(T):
        samples = np.zeros(N_e)
        for n in xrange(N_e):
            samples[n] = sl_chains[n][rand_idx[n,t], i]
        loc, scale = norm.fit(samples)
        gary_predictives[i, :] += norm.cdf(xs[i, :], loc=loc, scale=scale)
gary_predictives /= T
print gary_predictives

for i in xrange(num_gs):
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    ax.grid()
    ax.plot(xs[i, :], gary_predictives[i, :], lw=2, label="Predictive")
    ax.set_xlabel(r"$\log({})$".format(g_parameters[i]))
    ax.set_ylabel("CDF")
    ax.legend(loc=2)
    fig.tight_layout()
    plt.show()


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
        

        


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
if pyap_options["model_number"]==2:
    first_trace_number = int(split_trace_name[-1])
elif pyap_options["model_number"]==4:
    first_trace_number = int(split_trace_name[-2])
    
print first_trace_number
        
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)

g_labels = ["${}$".format(g) for g in g_parameters]

m_true = np.log(original_gs)
sigma2_true = 0.01


parallel = True


cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 101
nums_expts = [2, 4]#, 8, 16, 32]

labels = ("True",) + tuple(["$N_e = {}$".format(n) for n in nums_expts])
lines = ()

total_nums_expts = len(nums_expts)
color_idx = np.linspace(0, 1, total_nums_expts)

normpdf = norm.pdf
normrvs = norm.rvs
invgammarvs = invgamma.rvs



ax_titles = ['Single-level', 'Hierarchical']
fig = plt.figure(figsize=(7,10))


xs = []

for i in xrange(num_gs):
    ax1 = fig.add_subplot(4,2,2*i+1)
    ax2 = fig.add_subplot(4,2,2*i+2, sharex=ax1, sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax1.set_ylabel('Probability density')
    print "{} / {}\n".format(i+1, num_gs)
    xs.append(np.linspace(m_true[i]-2*np.sqrt(sigma2_true), m_true[i]+2*np.sqrt(sigma2_true), num_pts))
    x = xs[i]
    ax1.set_xlim(x[0], x[-1])
    for j, ax in enumerate([ax1, ax2]):
        ax.grid()
        if i==0:
            ax.set_title(ax_titles[j])
        ax.set_xlabel('log({})'.format(g_labels[i]), fontsize=16)
        line, = ax.plot(x, normpdf(x, loc=m_true[i], scale=np.sqrt(sigma2_true)), lw=2, color=cs[1], label="True")
        if i==0 and j==0:
            lines += (line,)
    for a, N_e in enumerate(nums_expts):
        colour = plt.cm.winter(color_idx[a])
        # Gary approximation
        sl_chains = []
        for n in xrange(N_e):
            if pyap_options["model_number"]==2:  # BR
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
            if t%100==0:
                print "samples:", samples
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
        if i==0:
            lines += (line,)
        ax1.plot(x, gary_pred, lw=2, color=colour, label="$N_e = {}$".format(N_e))
        ax2.plot(x, post_pred, lw=2, color=colour, label="$N_e = {}$".format(N_e))
        
        
    ax1.set_ylim(0, ax1.get_ylim()[1])
    xlim = ax1.get_xlim()
    xticks = ax1.get_xticks()
    if xlim[0] <= xticks[0]:
        ax1.set_xticks(xticks[1:-1])
    else:
        ax1.set_xticks(xticks[2:-1])
    new_ticks = ax1.get_xticks()
    if xlim[-1] >= new_ticks[-1]:
        ax1.set_xticks(new_ticks[:-1])
    else:
        ax1.set_xticks(new_ticks[:-2])
    
    
    for j, axx in enumerate([ax1, ax2]):
        axx.set_xlabel("log({})".format(g_labels[i]), fontsize=16)
        for tick in axx.get_xticklabels():
            tick.set_rotation(30)
       

leg = fig.legend(lines, labels, loc="upper center", ncol=1+len(nums_expts)/2, bbox_to_anchor=(0.5, 1.1))

fig.tight_layout()

fig_file = h_png_dir + "gary_predictive.png"
print fig_file
#fig.savefig(fig_file, bbox_extra_artists=(leg,), bbox_inches='tight', pad_inches=0.15)
plt.show()

        


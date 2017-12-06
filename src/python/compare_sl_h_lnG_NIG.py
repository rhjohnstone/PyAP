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


def new_eta(old_eta, samples): # for sampling from conjugate prior-ed N-IG
    x_bar = np.mean(samples)
    mu, nu, alpha, beta = 1.*old_eta
    new_mu = ((nu*mu + N_e*x_bar) / (nu + N_e))
    new_nu = nu + N_e
    new_alpha = alpha + 0.5*N_e
    new_beta = beta + 0.5*np.sum((samples-x_bar)**2) + 0.5*((N_e*nu)/(nu+N_e))*(x_bar-mu)**2
    return new_mu, new_nu, new_alpha, new_beta

    
randn = npr.randn
sqrt = np.sqrt
def sample_from_N_IG(eta):
    mu, nu, alpha, beta = eta
    sigma_squared_sample = invgamma.rvs(alpha,scale=beta)
    sample = mu + sqrt(sigma_squared_sample/nu)*randn()
    return sample, sigma_squared_sample
    

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

m_true = np.log(original_gs)
sigma2_true = 0.01

fig_y = 5
phi = 1.61803398875
figsize = (phi*fig_y, fig_y)

parallel = True

expt_params = np.loadtxt(expt_params_file)

cs = ['#1b9e77','#d95f02','#7570b3']
ax_titles = ["Single-level MLE pred.", "Hierarchical post. pred."]
num_pts = 101
nums_expts = [2, 4, 8, 16, 32]
total_nums_expts = len(nums_expts)
color_idx = np.linspace(0, 1, total_nums_expts)

normpdf = norm.pdf
normrvs = norm.rvs
invgammarvs = invgamma.rvs

#means = np.zeros((nums_expts[-1], num_gs))
#variances = np.zeros((nums_expts[-1], num_gs))

num_sl_samples = 100
sl_samples = np.zeros((num_sl_samples*nums_expts[-1], num_gs))

for n in xrange(nums_expts[-1]):
    temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(n)
    print "Trace:", temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    temp_chain = np.loadtxt(sl_mcmc_file, usecols=range(num_gs))
    #means[n, :] = temp_chain.mean(axis=0)
    #variances[n, :] = temp_chain.var(axis=0)
    sl_saved = temp_chain.shape[0]
    sl_idx = npr.randint(0, sl_saved, num_sl_samples)
    sl_samples[n*num_sl_samples:(n+1)*num_sl_samples, :] = temp_chain[sl_idx, :]
    
print sl_samples
sys.exit()

#print means
#print variances

for i in xrange(num_gs):
    print "{} / {}\n".format(i+1, num_gs+1)
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
    axs[0].set_ylabel('Probability density', fontsize=14)
    x = np.linspace(m_true[i]-2*np.sqrt(sigma2_true), m_true[i]+2*np.sqrt(sigma2_true), num_pts)
    for j in xrange(2):
        axs[j].grid()
        axs[j].set_title(ax_titles[j])
        axs[j].set_xlabel('log({})'.format(g_labels[i]), fontsize=14)
        axs[j].plot(x, normpdf(x, loc=m_true[i], scale=np.sqrt(sigma2_true)), lw=2, color=cs[1], label="True")
    for a, N_e in enumerate(nums_expts):
        colour = plt.cm.winter(color_idx[a])
        # MLE fit
        loc, scale = norm.fit(means[:N_e, i])
        alpha, _, beta = invgamma.fit(variances[:N_e, i], floc=0)
        print "invgamma for variances: alpha = {}, beta = {}".format(alpha, beta)
        # Posterior predictive
        hmcmc_file, log_file, h_png_dir, pdf_dir = ps.hierarchical_lnG_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
        h_chain = np.loadtxt(hmcmc_file,usecols=[i, num_gs+i])
        saved_its = h_chain.shape[0]
        start = time()
        mle_pred = np.zeros(num_pts)
        post_pred = np.zeros(num_pts)
        if args.num_samples == 0:
            T = saved_its
            mle_m = normrvs(loc=loc, scale=scale, size=T)
            mle_s2 = invgammarvs(alpha, loc=0, scale=beta, size=T)
            for t in xrange(T):
                post_pred += normpdf(x, loc=h_chain[t, 0], scale=np.sqrt(h_chain[t, 1]))
                mle_pred += normpdf(x, loc=mle_m[t], scale=np.sqrt(mle_s2[t]))
        else:
            T = args.num_samples
            rand_idx = npr.randint(0, saved_its, T)
            mle_m = normrvs(loc=loc, scale=scale, size=T)
            mle_s2 = invgammarvs(alpha, loc=0, scale=beta, size=T)
            for t in xrange(T):
                post_pred += normpdf(x, loc=h_chain[rand_idx[t], 0], scale=np.sqrt(h_chain[rand_idx[t], 1]))
                mle_pred += normpdf(x, loc=mle_m[t], scale=np.sqrt(mle_s2[t]))
        mle_pred /= T
        post_pred /= T
        tt = time()-start
        print "Time taken for MLE pred: {} s".format(round(tt))
        axs[0].plot(x, mle_pred, lw=2, color=colour, label="$N_e = {}$".format(N_e))
        axs[1].plot(x, post_pred, lw=2, color=colour, label="$N_e = {}$".format(N_e))

    for j in xrange(2):
        axs[j].legend(loc='best', fontsize=10)
            
    fig.tight_layout()
    #plt.show()
    mle_pred_dir = h_png_dir + "{}_sl_h_post_preds_N-IG/".format(expt_name)
    if not os.path.exists(mle_pred_dir):
        os.makedirs(mle_pred_dir)
    fig.savefig(mle_pred_dir+'sl_mle_pred_and_h_post_pred_N-IG_{}.png'.format(g_parameters[i]))
    plt.close()

        


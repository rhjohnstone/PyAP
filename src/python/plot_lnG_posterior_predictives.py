import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import norm
from scipy.stats import invgamma


#python_seed = 1
#npr.seed(python_seed)

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
requiredNamed.add_argument("-T", "--num-samples", type=int, help="number of samples to plot prior predictive from", required=True)

args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
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

parallel = True

T = args.num_samples

m_true = np.log(original_gs)
sigma2_true = 0.01

# BR only!

mu = m_true
alpha = 4.*np.ones(num_gs)
beta = (alpha+1.) * 0.04  # value used in hMCMC
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

old_eta_js = np.vstack((mu, nu, alpha, beta)).T

cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 101

ax_y = 6
phi = 1.61803398875
figsize = (phi*ax_y, ax_y)

if pyap_options["model_number"] == 2:
    figsize = (phi*ax_y, ax_y)
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=True, figsize=figsize)
elif pyap_options["model_number"] == 5:
    figsize = (2*phi*3, 10)
    fig, axs = plt.subplots(5, 3, sharex=False, sharey=True, figsize=figsize)
else:
    sys.exit("Only BR currently")
axs = axs.flatten()

if pyap_options["model_number"] == 5:
    axs[-1].axis('off')
    axs[-2].axis('off')

xs = []
for i in xrange(num_gs):
    axs[i].grid()
    xs.append(np.linspace(m_true[i]-2*np.sqrt(sigma2_true), m_true[i]+2*np.sqrt(sigma2_true), num_pts))
    x = xs[i]
    true, = axs[i].plot(x, norm.pdf(x, loc=m_true[i], scale=np.sqrt(sigma2_true)), lw=2, color=cs[1])
    prior_y = np.zeros(num_pts)
    for t in xrange(T):
        mean_sample, s2_sample = sample_from_N_IG(old_eta_js[i, :])
        prior_y += norm.pdf(x, loc=mean_sample, scale=np.sqrt(s2_sample))
    prior_y /= T
    prior, = axs[i].plot(x, prior_y, lw=2, color=cs[0])
    if i%2==0:
        axs[i].set_ylabel('Probability density')
    axs[i].set_xlabel('log({})'.format(g_labels[i]), fontsize=16)



nums_expts = [2,4]#,8,16,32]

labels = ("True", "Prior pred.") + tuple(["$N_e = {}$".format(n) for n in nums_expts])

color_idx = np.linspace(0, 1, len(nums_expts))
lines = ()
for j, N_e in enumerate(nums_expts):
    colour = plt.cm.winter(color_idx[j])
    mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_lnG_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
    h_chain = np.loadtxt(mcmc_file)
    saved_its, d = h_chain.shape
    for i in xrange(num_gs):
        x = xs[i]
        post_y = np.zeros(num_pts)
        idx = npr.randint(0, saved_its, T)
        for t in xrange(T):
            post_y += norm.pdf(x, loc=h_chain[idx[t],i], scale=np.sqrt(h_chain[idx[t],num_gs+i]))
        post_y /= T
        post, = axs[i].plot(x, post_y, lw=2, color=colour)
        if i==0:
            lines += (post,)
        
for i in xrange(num_gs):
    axs[i].set_xlim(xs[i][0], xs[i][-1])
    axs[i].set_ylim(0, axs[i].get_ylim()[1])
    
print (true, prior)+lines
print labels
fig.legend((true, prior)+lines, labels, loc=9, mode="expand", ncol=2+len(nums_expts), bbox_to_anchor=(0., 1.02, 1., .102))

fig.tight_layout()

#fig.savefig(png_dir + "{}_{}_traces_hMCMC_post_preds.png".format(expt_name,N_e))

plt.show()




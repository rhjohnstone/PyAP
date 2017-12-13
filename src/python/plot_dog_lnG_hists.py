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
requiredNamed.add_argument("-n", "--num-traces", type=int, help="which hMCMC to use", required=True)

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

    
N_e = args.num_traces

color_idx = np.linspace(0, 1, N_e)


m_true = np.log(original_gs)
sigma2_true = 0.01

sigma_lower = 1e-3
sigma_upper = 25

sigma_const = 1./(sigma_upper-sigma_lower)

mu = m_true
alpha = 4.*np.ones(num_gs)
beta = (alpha+1.) * sigma2_true
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

prior_means = np.log(original_gs)
prior_sd = 0.5*np.log(10)

old_eta_js = np.vstack((mu, nu, alpha, beta)).T

cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 101

ax_x = 8
phi = 1.61803398875
golden = (ax_x, phi*ax_x)
a4 = (ax_x, np.sqrt(2)*ax_x)

figsize = (ax_x, 1.2*ax_x)

fig, axs = plt.subplots(5, 3, figsize=a4)
axs = axs.flatten()
for i in xrange(num_gs+1):
    axs[i].grid()
    if i < num_gs:
        axs[i].set_xlabel('log({})'.format(g_labels[i]), fontsize=16)
    else:
        axs[i].set_xlabel(r"$\sigma$")
    if i%3==0:
        axs[i].set_ylabel('Norm. freq.')

sl_means = np.zeros((N_e, num_gs))
for n in xrange(N_e):
    temp_trace_number = first_trace_number + n
    temp_trace_name = "_".join(split_trace_name[:-1])+"_"+str(temp_trace_number)
    print temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    sl_chain = np.loadtxt(sl_mcmc_file)
    sl_means[n, :] = np.mean(sl_chain[:,:-2], axis=0)
    
    colour = plt.cm.winter(color_idx[n])
    c = matplotlib.colors.colorConverter.to_rgba(colour, alpha=1.5/N_e)
    for i in xrange(num_gs+1):
        axs[i].hist(sl_chain[:, i], normed=True, color=c, lw=0, bins=40, zorder=10)

num_ticks = 5
for i in xrange(num_gs+1):
    xlim = axs[i].get_xlim()
    xdiff = xlim[1]-xlim[0]
    lower_x = xlim[0] + 0.2*xdiff
    mid_x = xlim[0] + 0.5*xdiff
    upper_x = xlim[0] + 0.8*xdiff
    axs[i].set_xticks(np.round([lower_x, mid_x, upper_x],3))
    ylim = axs[i].get_ylim()
    axs[i].set_yticks(np.round(np.linspace(0, ylim[1], num_ticks),2))
    plt.setp( axs[i].xaxis.get_majorticklabels(), rotation=30 )
    x = np.linspace(xlim[0], xlim[1], num_pts)
    axprior = axs[i].twinx()
    if i%3==2:
        axprior.set_ylabel("Prob. dens.")
    if i<num_gs:
        axprior.plot(x, norm.pdf(x, loc=prior_means[i], scale=prior_sd), lw=2, color=cs[1], zorder=0)
    else:
        axprior.axhline(sigma_const, lw=2, color=cs[1], zorder=0)

fig.tight_layout()#h_pad=1.)
fig_file = sl_png_dir+"{}_{}_traces_superimposed_marginal_hists.png".format(expt_name, N_e)
print fig_file
fig.savefig(fig_file)
#plt.show()
sys.exit()

for i in xrange(num_gs):
    fig = plt.figure(figsize=(6,4))
    
    indices = np.arange(2*num_gs + i, (2+N_e)*num_gs + i, num_gs)
    means = np.mean(chain[:, indices], axis=0)
    x = np.linspace(np.min(means)-0.5, np.max(means)+0.5, num_pts)
    print "expt-level means:", means
    loc, scale = norm.fit(means)
    print "best fit loc: {}, scale: {}".format(loc, scale)
    
    axpp = fig.add_subplot(111)
    axpp.grid()
    prior_y = np.zeros(num_pts)
    post_y = np.zeros(num_pts)
    axpp.plot(means, np.zeros(N_e), 'x', color=cs[0], ms=10, mew=2, label='Expt means', clip_on=False, zorder=10)
    for t in xrange(T):
        mean_sample, s2_sample = sample_from_N_IG(old_eta_js[i, :])
        prior_y += norm.pdf(x, loc=mean_sample, scale=np.sqrt(s2_sample))
        idx = npr.randint(saved_its)
        post_y += norm.pdf(x, loc=chain[idx,i], scale=np.sqrt(chain[idx,num_gs+i]))
    prior_y /= T
    post_y /= T
    axpp.plot(x, norm.pdf(x, loc=loc, scale=scale), lw=2, color=cs[0], label='MLE fit')
    axpp.plot(x, post_y, lw=2, color=cs[2], label='Post. pred.')
    axpp.plot(x, norm.pdf(x, loc=m_true[i], scale=np.sqrt(sigma2_true)), label='True', lw=2, color=cs[1])
    ylim = axpp.get_ylim()
    axpp.set_ylim(0, ylim[1])
    axpp.set_ylabel('Probability density')
    axpp.set_xlabel('log({})'.format(g_labels[i]), fontsize=16)
    axpp.legend(loc='best')
    #axpp.set_yticks(np.linspace(axpp.get_yticks()[0],axpp.get_yticks()[-1],len(ax2.get_yticks())))


    fig.tight_layout()
    #fig.savefig(png_dir+"{}_{}_traces_hierarchical_{}_posterior_predictive.png".format(expt_name, N_e, g_parameters[i]))
    plt.close()




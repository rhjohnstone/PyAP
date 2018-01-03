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
requiredNamed.add_argument("-x", "--num-pts", type=int, help="how many x points to plot Gary-predictive for", required=True)
parser.add_argument("-T", "--num-samples", type=int, help="number of samples to plot prior predictive from", default=0)

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


if expt_name=="dog_teun_davies":
    first_trace_number = 150
elif expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
    first_trace_number = int(split_trace_name[-2])

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

if expt_name=="dog_teun_davies":
    fig, axs = plt.subplots(5, 3, figsize=figsize)
elif expt_name=="roche_ten_tusscher_correct_units":
    fig, axs = plt.subplots(4, 3, figsize=figsize)
elif expt_name=="roche_paci_correct_units":
    fig, axs = plt.subplots(4, 3, figsize=figsize)

axs = axs.flatten()
for i in xrange(num_gs+1):
    axs[i].grid()
    if i < num_gs:
        axs[i].set_xlabel('log({})'.format(g_labels[i]), fontsize=15)
    else:
        axs[i].set_xlabel(r"$\sigma$", fontsize=15)
    if i%3==0:
        axs[i].set_ylabel('Norm. freq.')

for n in xrange(N_e):
    temp_trace_number = first_trace_number + n
    if expt_name=="dog_teun_davies":
        temp_trace_name = "_".join(split_trace_name[:-1])+"_"+str(temp_trace_number)
    elif expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
        temp_trace_name = "_".join(split_trace_name[:-2])+"_{}_1".format(temp_trace_number)
    print temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    try:
        sl_chain = np.loadtxt(sl_mcmc_file)
        if expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
            saved_its = sl_chain.shape[0]
            sl_chain = sl_chain[saved_its/2:, :]
    except:
        print "Can't load", sl_mcmc_file

    
    colour = plt.cm.winter(color_idx[n])
    c = matplotlib.colors.colorConverter.to_rgba(colour, alpha=3./N_e)
    for i in xrange(num_gs+1):
        axs[i].hist(sl_chain[:, i], normed=True, color=c, lw=0, bins=40)

num_ticks = 5
axpriors = []
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
    axpriors.append(axs[i].twinx())
    axprior = axpriors[i]
    axprior.axes.get_yaxis().set_visible(False)
    if i<num_gs:
        axprior.set_ylim(0, 0.35)
        axprior.plot(x, norm.pdf(x, loc=prior_means[i], scale=prior_sd), "--", lw=2, color=cs[1], alpha=0.5)
    else:
        axprior.set_ylim(0, 2*sigma_const)
        axprior.axhline(sigma_const, linestyle="--", lw=2, color=cs[1], alpha=0.5)
    


#fig.savefig(fig_file)
#plt.show()



normpdf = norm.pdf
normrvs = norm.rvs

N_e = args.num_traces

xs = []
sl_chains = []
mins = 1e9*np.ones(num_gs+1)
maxs = -1e9*np.ones(num_gs+1)
chain_lengths = []
for n in xrange(N_e):
    #if (pyap_options["model_number"]==2) or (pyap_options["model_number"]==5):  # synth BR/OH
    #    temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(n)
    #elif (pyap_options["model_number"]==2) or (pyap_options["model_number"]==6):  # Davies
    #    temp_trace_name = "_".join(split_trace_name[:-1]) + "_" + str(150+n)
    if expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
        temp_trace_name = "_".join(split_trace_name[:-2]) + "_{}_".format(100+n) + split_trace_name[-1]
    elif expt_name=="dog_teun_davies":
        temp_trace_name = "_".join(split_trace_name[:-1]) + "_{}".format(150+n)
    print "Trace:", temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    try:
        temp_chain = np.loadtxt(sl_mcmc_file, usecols=range(num_gs+1))
    except:
        print "Can't load", sl_mcmc_file
        continue
    saved_its = temp_chain.shape[0]
    if expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
        temp_chain = temp_chain[(3*saved_its)/5:, :]
    elif expt_name=="dog_teun_davies":
        temp_chain = temp_chain[saved_its/3:, :]
    sl_chains.append(temp_chain)
    temp_mins = np.min(temp_chain, axis=0)
    temp_maxs = np.max(temp_chain, axis=0)
    where_new_min = np.where(temp_mins < mins)
    where_new_max = np.where(temp_maxs > maxs)
    mins[where_new_min] = temp_mins[where_new_min]
    maxs[where_new_max] = temp_maxs[where_new_max]
    chain_lengths.append(temp_chain.shape[0])
    
length = chain_lengths[0]
print length
print chain_lengths
assert(np.all(np.array(chain_lengths)==length))

N_e = len(chain_lengths)

num_pts = args.num_pts
xs = np.zeros((num_gs+1, num_pts))
for i in xrange(num_gs+1):
    hist_xlim = axs[i].get_xlim()
    if expt_name=="roche_ten_tusscher_correct_units":
        if i==0 or i==1 or i==3:
            xs[i, :] = np.linspace(mins[i]-0.2, maxs[i]+0.2, num_pts)
        else:
            xs[i, :] = np.linspace(mins[i]-1.5, maxs[i]+1.5, num_pts)
    elif expt_name=="roche_paci_correct_units":
        if i==0 or i==3:
            xs[i, :] = np.linspace(mins[i]-0.5, maxs[i]+0.5, num_pts)
        else:
            xs[i, :] = np.linspace(mins[i]-4., maxs[i]+6., num_pts)
    else:
        xs[i, :] = np.linspace(hist_xlim[0], hist_xlim[1], num_pts)


T = args.num_samples
gary_predictives = np.zeros((num_gs+1, num_pts))
rand_idx = npr.randint(0, length, size=(N_e, T))  # don't know if this will cause memory/speed issues
for i in xrange(num_gs+1):
    for t in xrange(T):
        samples = np.zeros(N_e)
        for n in xrange(N_e):
            samples[n] = sl_chains[n][rand_idx[n,t], i]
        loc, scale = norm.fit(samples)
        gary_predictives[i, :] += norm.pdf(xs[i, :], loc=loc, scale=scale)
gary_predictives /= T

pred_colour = "#e7298a"
for k in xrange(num_gs+1):
    axs[k].plot(xs[k], gary_predictives[k], color=pred_colour)
    
    
fig.tight_layout()#h_pad=1.)
fig_file = sl_png_dir+"{}_{}_traces_superimposed_marginal_hists_and_prior_and_gary_predictive.png".format(expt_name, N_e)
print fig_file
fig.savefig(fig_file)
plt.show(block=True)



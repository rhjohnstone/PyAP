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
requiredNamed.add_argument("-T", "--num-samples", type=int, help="number of samples to plot prior predictive from", required=True)
parser.add_argument("--different", action="store_true", help="use different initial guess for some params", default=False)

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


"""for n, i in it.product(range(2), range(num_gs)):
    print n, i
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    if n == 0:
        xlabel = r"\hat{" + g_parameters[i] +"}$"
        figname = "trace_{}_top_{}.png".format(n, g_parameters[i])
    elif n == 1:
        xlabel = r"\sigma_{" + g_parameters[i] +"}^2$"  # need to check if this squared is correct
        figname = "trace_{}_sigma_{}_squared.png".format(n, g_parameters[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised frequency")
    idx = (2+n)*num_gs + i
    ax.hist(chain[burn:, idx], bins=40, color='blue', edgecolor='blue')"""
    #fig.savefig(png_dir+figname)
    #plt.close()

    
N_e = args.num_traces

expt_params = np.loadtxt(expt_params_file)[:N_e,:]

color_idx = np.linspace(0, 1, N_e)
mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_lnG_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
h_chain = np.loadtxt(mcmc_file)
saved_its, d = h_chain.shape

T = args.num_samples

best_fits_params = np.zeros((args.num_traces, num_gs))
for i in xrange(args.num_traces):
    best_params = np.loadtxt('/'.join( split_trace_path[:5] ) + "/expt_params.txt")[i, :]
starting_points = np.copy(best_fits_params)
starting_mean = np.mean(starting_points,axis=0)
starting_vars = np.var(starting_points,axis=0)

m_true = np.log(original_gs)
sigma2_true = 0.01

mu = m_true
alpha = 4.*np.ones(num_gs)
beta = (alpha+1.) * sigma2_true
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

old_eta_js = np.vstack((mu, nu, alpha, beta)).T

cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 201

ax_y = 3
phi = 1.61803398875
figsize = (phi*ax_y, 2*ax_y)
g_figs = []
g_axs = []
bottoms_axs = []
titles = ["Hierarchical", "Single-level"]
for i in xrange(num_gs):
    x = np.linspace(m_true[i]-2*np.sqrt(sigma2_true), m_true[i]+2*np.sqrt(sigma2_true), num_pts)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    g_figs.append(fig)
    g_axs.append((axs[0].twinx(), axs[1].twinx()))
    bottoms_axs.append(axs)
    for j in xrange(2):
        axs[j].yaxis.set_label_position("right")
        axs[j].yaxis.tick_right()
        axs[j].grid()
        axs[j].set_title(titles[j])
        axs[j].set_ylabel('Probability density')
        axs[j].plot(x, norm.pdf(x, loc=m_true[i], scale=np.sqrt(sigma2_true)), label='True', lw=2, color=cs[1])
        
        g_axs[i][j].yaxis.set_label_position("left")
        g_axs[i][j].yaxis.tick_left()
        g_axs[i][j].set_ylabel('Normalised frequency')
    axs[1].set_xlabel('log({})'.format(g_labels[i]))

sl_means = np.zeros((N_e, num_gs))
for n in xrange(N_e):
    temp_trace_name = "_".join(split_trace_name[:-1])+"_"+str(n)
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    sl_chain = np.loadtxt(sl_mcmc_file)
    sl_means[n, :] = np.mean(sl_chain[:,:-2], axis=0)
    
    colour = plt.cm.winter(color_idx[n])
    c = matplotlib.colors.colorConverter.to_rgba(colour, alpha=2./N_e)
    for i in xrange(num_gs):
        h_expt_idx = (2+n)*num_gs + i
        g_axs[i][0].hist(h_chain[:, h_expt_idx], normed=True, color=c, lw=0, bins=40)
        g_axs[i][1].hist(sl_chain[:, i], normed=True, color=c, lw=0, bins=40)
        for j in xrange(2):
            g_axs[i][j].plot(np.log(expt_params[n, i]), 0, 'x', color=cs[1], ms=10, mew=2, clip_on=False, zorder=10)

for i in xrange(num_gs):
    x = np.linspace(m_true[i]-2*np.sqrt(sigma2_true), m_true[i]+2*np.sqrt(sigma2_true), num_pts)
    post_y = np.zeros(num_pts)
    for t in xrange(T):
        idx = npr.randint(saved_its)
        post_y += norm.pdf(x, loc=h_chain[idx,i], scale=np.sqrt(h_chain[idx,num_gs+i]))
    post_y /= T
    bottoms_axs[i][0].plot(x, post_y, lw=2, color=cs[2], label='Post. pred.')
    loc, scale = norm.fit(sl_means[:, i])
    bottoms_axs[i][1].plot(x, norm.pdf(x,loc=loc, scale=scale), lw=2, color=cs[0], label='MLE fit')
    bottoms_axs[i][0].legend(loc='best')
    bottoms_axs[i][1].legend(loc='best')

plt.show()
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
    axpp.set_xlabel('log({})'.format(g_labels[i]))
    axpp.legend(loc='best')
    #axpp.set_yticks(np.linspace(axpp.get_yticks()[0],axpp.get_yticks()[-1],len(ax2.get_yticks())))


    fig.tight_layout()
    #fig.savefig(png_dir+"{}_{}_traces_hierarchical_{}_posterior_predictive.png".format(expt_name, N_e, g_parameters[i]))
    plt.close()




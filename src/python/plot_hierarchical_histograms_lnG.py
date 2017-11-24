import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
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
chain = np.loadtxt(mcmc_file)
saved_its, d = chain.shape
chain = chain[saved_its/2:, :]
saved_its, d = chain.shape

T = args.num_samples

best_fits_params = np.zeros((args.num_traces, num_gs))
for i in xrange(args.num_traces):
    best_params = np.loadtxt('/'.join( split_trace_path[:5] ) + "/expt_params.txt")[i, :]
    if args.different: # 9,10,11
        for j in [9,10,11]:
            best_params[j] = 10 * original_gs[j] * npr.rand()
    best_fits_params[i, :] = np.copy(best_params)
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
for i in xrange(num_gs):
    fig = plt.figure(figsize=(8,6))
    ax2 = fig.add_subplot(111)
    ax2.grid()
    ax2.set_xlabel("log({})".format(g_labels[i]))
    ax2.set_ylabel("Normalised frequency")
    #ax2.axvline(original_gs[i], color='green', lw=2, label='true top')
    #xmin = 1e9
    #xmax = -1e9
    for n in xrange(N_e):
        idx = (2+n)*num_gs + i
        colour = plt.cm.winter(color_idx[n])
        ax2.hist(chain[:, idx], normed=True, bins=40, color=colour, edgecolor=None, alpha=2./N_e)
        ax2.axvline(np.log(expt_params[n, i]), color='red', lw=2)
        temp_min = np.min(chain[:, idx])
        temp_max = np.max(chain[:, idx])
        #if temp_min < xmin:
        #    xmin = temp_min
        #if temp_max > xmax:
        #    xmax = temp_max
    plt.xticks(rotation=30)
    
    axpp = ax2.twinx()
    axpp.grid()
    xlim = ax2.get_xlim()
    x = np.linspace(xlim[0]-0.2, xlim[1]+0.2, num_pts)
    prior_y = np.zeros(num_pts)
    post_y = np.zeros(num_pts)
    for t in xrange(T):
        mean_sample, s2_sample = sample_from_N_IG(old_eta_js[i, :])
        prior_y += norm.pdf(x, loc=mean_sample, scale=np.sqrt(s2_sample))
        idx = npr.randint(saved_its)
        post_y += norm.pdf(x, loc=chain[idx,i], scale=np.sqrt(chain[idx,num_gs+i]))
    prior_y /= T
    post_y /= T
    axpp.plot(x, prior_y, lw=2, color=cs[0], label='Prior pred.')
    axpp.plot(x, post_y, lw=2, color=cs[1], label='Post. pred.')
    axpp.plot(x, norm.pdf(x, loc=m_true[i], scale=np.sqrt(sigma2_true)), label='True', lw=2, color=cs[2])
    ylim = axpp.get_ylim()
    axpp.set_ylim(0, ylim[1])
    axpp.set_ylabel('Probability density')
    axpp.legend()
    axpp.set_yticks(np.linspace(axpp.get_yticks()[0],axpp.get_yticks()[-1],len(ax2.get_yticks())))


    fig.tight_layout()
    fig.savefig(png_dir+"{}_{}_traces_hierarchical_{}_marginal.png".format(expt_name, N_e, g_parameters[i]))
    plt.close()

sys.exit()
for i in xrange(num_gs):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_xlabel("sigma squared "+g_labels[i])
    ax.set_ylabel("Normalised frequency")
    ax.hist(chain[:, num_gs+i], normed=True, bins=40, color='blue', edgecolor='blue')
    ax.axvline((0.2*original_gs[i])**2, color='red', lw=2)  # just for synthetic OH atm
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(png_dir+"{}_{}_traces_hierarchical_sigma2_{}_marginal.png".format(expt_name, N_e, g_parameters[i]))
    plt.close()

#plt.show(block=True)


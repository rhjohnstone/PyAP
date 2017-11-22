#import ap_simulator
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, lognorm

start = time.time()

inf = np.inf
dot = np.dot
multivariate_normal = npr.multivariate_normal
npcopy = np.copy
exp = np.exp
sqrt = np.sqrt
npsum = np.sum
nplog = np.log
nprrandn = npr.randn

def solve_for_voltage_trace(temp_g_params, ap_model_index):
    if np.any(temp_g_params<0):
        return np.zeros(num_pts)
    ap_models[ap_model_index].SetToModelInitialConditions()
    try:
        return ap_models[ap_model_index].SolveForVoltageTraceWithParams(temp_g_params)
    except:
        print "Failed to solve"
        print "temp_g_params:", temp_g_params
        return np.zeros(num_pts)
        

def solve_star(temp_g_params_and_ap_model_index):
    return solve_for_voltage_trace(*temp_g_params_and_ap_model_index)
    

python_seed = 1
npr.seed(python_seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("--num-traces", type=int, help="number of traces to fit to, including the one specified as argument", required=True)
requiredNamed.add_argument("-T", "--num-samples", type=int, help="number of samples to construct prior and posterior predictive distributions", required=True)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
args, unknown = parser.parse_known_args()

N_e = args.num_traces

trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"
expt_params_file = '/'.join( split_trace_path[:5] ) + "/{}expt_params.txt".format(expt_dir)

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

expt_params = np.loadtxt(expt_params_file)[:N_e, :]
print "expt_params:\n", expt_params

split_trace_name = trace_name.split("_")
first_trace_number = int(split_trace_name[-1])  # need a specific-ish format currently
trace_numbers = range(first_trace_number, first_trace_number+N_e)


parallel = True

mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)

initial_it_file = log_file[:-4]+"_initial_iteration.txt"


gamma_hyperparams = np.zeros((num_gs,2))
gamma_hyperparams[:,0] = 10.  # alpha
gamma_hyperparams[:,1] = 0.25 * np.log(10)**2 * (gamma_hyperparams[:,0]-1.)  # beta

normal_hyperparams = np.zeros((num_gs,2))
normal_hyperparams[:,0] = np.log(original_gs) + 0.25 * np.log(10)**2  # s
normal_hyperparams[:,1] = 4.  # lambda


def update_normal_hyperparams(old_hypers, nu_hat, samples):
    s, t = old_hypers
    new_s = (nu_hat * npsum(np.log(samples)) + t*s) / (N_e * nu_hat + t)
    new_t = N_e * nu_hat + t
    return new_s, new_t


def sample_from_updated_top_mu_normal(new_s, new_t):
    mu_sample = norm.rvs(loc=new_s, scale=sqrt(new_t))
    return mu_sample


def update_gamma_hyperparams(old_hypers, g_hat, samples):
    alpha, beta = old_hypers
    new_alpha = alpha + N_e/2.
    new_beta = beta + 0.5*npsum((np.log(samples)-g_hat)**2)
    return new_alpha, new_beta


def sample_from_updated_top_tau_gamma(new_alpha, new_beta):
    tau_sample = gamma.rvs(new_alpha, scale=1./new_beta)
    return tau_sample



#sys.exit()

uniform_noise_prior = [0.,25.]
    
    
def log_pi_g_i(g_i, mus, taus, sigma, data_i, test_i):
    if np.any(g_i < 0) or (not (uniform_noise_prior[0] < sigma < uniform_noise_prior[1])):
        return -inf
    else:
        return -npsum((data_i - test_i)**2)/(2.*sigma**2) - npsum(nplog(g_i)) - npsum(taus*(nplog(g_i)-mus)**2 / 2.)


def log_pi_sigma(expt_datas, test_datas, sigma):
    if (not (uniform_noise_prior[0] < sigma < uniform_noise_prior[1])):
        return -inf
    else:
        return -N_e*num_pts*np.log(sigma) - npsum((expt_datas-test_datas)**2) / (2*sigma**2)


def compute_initial_sigma(expt_datas, test_datas):
    return sqrt(npsum((expt_datas-test_datas)**2) / (N_e*num_pts))


s = 0.2
mu = np.log(original_gs) - s**2/2
tau = np.ones(num_gs)/np.sqrt(s)






chain = np.loadtxt(mcmc_file)
saved_its, d = chain.shape

T = args.num_samples

color_idx = np.linspace(0, 1, N_e)
cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 501
top_colour = 'blue'
for i in xrange(num_gs):
    #mu_fig = plt.figure(figsize=(8,6))
    #mu_ax = mu_fig.add_subplot(111)
    #mu_ax.grid()
    #mu_ax.hist(chain[:, i], normed=True, bins=40, color=top_colour, edgecolor=top_colour)
    #plt.show()
    
    
    
    fig = plt.figure(figsize=(8,6))
    ax2 = fig.add_subplot(111)
    ax2.grid()
    ax2.set_xlabel("log10($"+g_parameters[i]+"$)")
    ax2.set_ylabel("Normalised frequency")
    #ax2.axvline(original_gs[i], color='green', lw=2, label='true top')
    xmin, xmax = 1e9, -1e9
    for n in xrange(N_e):
        idx = (2+n)*num_gs + i
        colour = plt.cm.winter(color_idx[n])
        ax2.hist(np.log10(chain[:, idx]), normed=True, bins=40, color=colour, edgecolor=None, alpha=2./N_e)
        temp_min = np.min(chain[:, idx])
        temp_max = np.max(chain[:, idx])
        if temp_min < xmin:
            xmin = temp_min
        if temp_max > xmax:
            xmax = temp_max
        ax2.axvline(np.log(expt_params[n, i]), color='red', lw=2)
    plt.xticks(rotation=30)
    
    x = np.logspace(np.log10(xmin)-4, np.log10(xmax)+4, num_pts)
    prior_y = np.zeros(num_pts)
    post_y = np.zeros(num_pts)
    prior_mus = []
    post_mus = []
    for _ in xrange(T):
        mu = norm.rvs(loc=normal_hyperparams[i,0], scale=1./np.sqrt(normal_hyperparams[i,1]))
        tau = gamma.rvs(gamma_hyperparams[i,0], scale=1./gamma_hyperparams[i,1])
        prior_mus.append(mu)
        prior_y += lognorm.pdf(x, s=1./np.sqrt(tau), scale=np.exp(mu))
        
        rand_idx = npr.randint(saved_its)
        mu_sample, tau_sample = chain[rand_idx, [i, num_gs+i]]
        post_mus.append(mu_sample)
        post_y += lognorm.pdf(x, s=1./np.sqrt(tau_sample), scale=np.exp(mu_sample))
        
    prior_y /= T
    post_y /= T
    ax2.set_ylabel("Probability density")
    #ax2.plot(x, prior_y, lw=2, color=cs[0], label="Prior pred.")
    #ax2.plot(x, post_y, lw=2, color=cs[1], label="Post. pred.")
    ax2.legend(loc=2)
    
    figg, (axx1, axx2) = plt.subplots(1,2)
    axx1.hist(prior_mus)
    axx2.hist(post_mus)    
    #axpp = ax2.twinx()
    #axpp.grid()
    #xlim = ax2.get_xlim()
    #x = np.linspace(0.8*xlim[0], 1.2*xlim[1], num_pts)
    #post_y = np.zeros(num_pts)
    #prior_y = np.zeros(num_pts)
    #for t in xrange(T):
    #    post_y += norm.pdf(x, loc=chain[t,i], scale=np.sqrt(chain[t,num_gs+i]))/(1.-norm.cdf(0, loc=chain[t,i], scale=np.sqrt(chain[t,num_gs+i])))  # scale for truncating at 0
    #    mean_sample, s2_sample = sample_from_N_IG(old_eta_js[i, :])
    #    prior_y += norm.pdf(x, loc=mean_sample, scale=np.sqrt(s2_sample))/(1.-norm.cdf(0, loc=mean_sample, scale=np.sqrt(s2_sample)))  # scale for truncating at 0
    #post_y /= T
    #prior_y /= T
    #axpp.plot(x, post_y, lw=2, color=cs[0], label='Post. pred.')
    #axpp.plot(x, prior_y, lw=2, color=cs[1], label='Prior pred.')
    #ylim = axpp.get_ylim()
    #axpp.set_ylim(0, ylim[1])
    #axpp.set_ylabel('Probability density')
    #axpp.legend()
    #axpp.set_yticks(np.linspace(axpp.get_yticks()[0],axpp.get_yticks()[-1],len(ax2.get_yticks())))


    fig.tight_layout()
    #fig.savefig(png_dir+"{}_{}_traces_hierarchical_{}_marginal.png".format(expt_name, N_e, g_parameters[i]))
    plt.show()



















#import ap_simulator
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm
from multiprocessing import Pool

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


split_trace_name = trace_name.split("_")
first_trace_number = int(split_trace_name[-1])  # need a specific-ish format currently
trace_numbers = range(first_trace_number, first_trace_number+N_e)


parallel = True

mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)

initial_it_file = log_file[:-4]+"_initial_iteration.txt"

coeffs = [1, -1, 0, 0, -16]
roots = np.roots(coeffs)
assert(np.isreal(roots[0]))
v = float(roots[0])  # manually ascertained

gamma_hyperparams = np.zeros((num_gs,2))
gamma_hyperparams[:,0] = 3.  # alpha
gamma_hyperparams[:,1] = (gamma_hyperparams[:,0]-1.)*np.log(v)  # beta

normal_hyperparams = np.zeros((num_gs,2))
normal_hyperparams[:,0] = np.log(original_gs) + gamma_hyperparams[:,1] / (gamma_hyperparams[:,0]-1.)  # s
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

color_idx = np.linspace(0, 1, N_e)
cs = ['#1b9e77','#d95f02','#7570b3']
num_pts = 201
for i in xrange(num_gs):
    fig = plt.figure(figsize=(8,6))
    ax2 = fig.add_subplot(111)
    ax2.grid()
    ax2.set_xlabel("log10($"+g_parameters[i]+"$)")
    ax2.set_ylabel("Normalised frequency")
    #ax2.axvline(original_gs[i], color='green', lw=2, label='true top')
    for n in xrange(N_e):
        idx = (2+n)*num_gs + i
        colour = plt.cm.winter(color_idx[n])
        ax2.hist(np.log10(chain[:, idx]), normed=True, bins=40, color=colour, edgecolor=None, alpha=2./N_e)
        #ax2.axvline(expt_params[n, i], color='red', lw=2)
    plt.xticks(rotation=30)
    
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



















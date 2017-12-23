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
beta = (alpha+1.) * 0.04
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

prior_means = np.log(original_gs)
prior_sd = 0.5*np.log(10)

old_eta_js = np.vstack((mu, nu, alpha, beta)).T



MPDs = np.zeros((N_e, num_gs))
for n in xrange(N_e):
    temp_trace_number = first_trace_number + n
    temp_trace_name = "_".join(split_trace_name[:-1])+"_"+str(temp_trace_number)
    print temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    sl_chain = np.loadtxt(sl_mcmc_file)
    max_target_idx = np.argmax(sl_chain[:,-1])
    MPDs[n, :] = sl_chain[max_target_idx, :-2]

dp = 3
means = MPDs.mean(axis=0).round(dp)
stds = MPDs.std(axis=0).round(dp)

print "\n"
for i in xrange(num_gs):
    line = r"${}$ & {} & {} \\".format(g_parameters[i], means[i], stds[i])
    print line
print "\n"









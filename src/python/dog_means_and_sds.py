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
parser.add_argument("--dp", type=int, help="how many decimal places to round means and stds to", default=2)

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

parameters = g_parameters + [r"\sigma"]
labels = ["${}$".format(p) for p in parameters]

N_e = args.num_traces

print "\n"
dp = args.dp
means_stds_weaved = np.zeros(2*(num_gs+1))
all_stuff = np.zeros((2*(num_gs+1), N_e))

pre_first_line = r"\begin{tabular}{*{"+str(N_e+1)+"}{c|}}"
print pre_first_line

first_line = " & " + " & ".join(str(first_trace_number + n) for n in xrange(N_e)) + r" \\"

print first_line
print r"\midrule"

# work out stuff, but no LaTeX here
for n in xrange(N_e):
    temp_trace_number = first_trace_number + n
    temp_trace_name = "_".join(split_trace_name[:-1])+"_"+str(temp_trace_number)
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    sl_chain = np.loadtxt(sl_mcmc_file, usecols=range(num_gs+1))
    means = sl_chain.mean(axis=0)
    stds = sl_chain.std(axis=0)
    means_stds_weaved[0::2] = means
    means_stds_weaved[1::2] = stds
    all_stuff[:, n] = means_stds_weaved
    
rounded_all_stuff = all_stuff.round(dp)

# now do LaTeX
for i in xrange(2*(num_gs+1)):
    p = i/2
    if i%2==0:
        line = r"\multirow{2}{*}{" + parameters[p] + "}"
    else:
        line = ""
    for n in xrange(N_e):
        line += " & " + str(rounded_all_stuff[i,n])
    line += r" \\"
    print line

print r"\midrule"

fssh

all_means = all_stuff.mean(axis=0)
all_stds = all_stuff.std(axis=0)
final_line = np.zeros(4*(num_gs+1))
final_line[0::2] = all_means
final_line[1::2] = all_stds
final_latex = " & ".join([str(x) for x in final_line.round(dp)])
print final_latex, r"\\"
print r"\bottomrule"

print "\n"
    


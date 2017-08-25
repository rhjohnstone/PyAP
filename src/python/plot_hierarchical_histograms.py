import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import matplotlib.pyplot as plt


#python_seed = 1
#npr.seed(python_seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("--num-traces", type=int, help="number of traces to fit to, including the one specified as argument", required=True)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-p', '--parallel', action='store_true', help="plot hMCMC that was run in parallel")
group.add_argument('-s', '--series', action='store_true', help="plot hMCMC that was run in series")

parser.add_argument("-nc", "--num-cores", type=int, help="number of cores to parallelise solving expt traces", default=1)
#parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
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



mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, args.num_traces, args.parallel)

chain = np.loadtxt(mcmc_file)
num_saved_its, num_params = chain.shape
burn = num_saved_its/4

plt.hist(chain[burn:, 3*num_gs], bins=40, color='blue', edgecolor='blue')
plt.show(block=True)



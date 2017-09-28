import numpy as np
import numpy.random as npr
import scipy.stats as st
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import pyap_setup as ps
import argparse


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
#requiredNamed.add_argument("--num-traces", type=int, help="number of traces to fit to, including the one specified as argument", required=True)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
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

true_params = np.loadtxt("projects/PyAP/python/input/{}/expt_params.txt".format(expt_name))


colors = ['#ffffb2','#fed976','#feb24c','#fd8d3c','#f03b20','#bd0026']


model = pyap_options["model_number"]

if model==1:
    label="hodgkin_huxley"
    expt_params_normal_sd = 0.15
elif model==2:
    label = "beeler_reuter"
    expt_params_normal_sd = 0.15
elif model==3:
    label = "luo_rudy"
    expt_params_normal_sd = 0.1
elif model==4:
    label = "ten_tusscher"
    expt_params_normal_sd = 0.3
elif model==5:
    label = "ohara"
    expt_params_normal_sd = 0.2
elif model==6:
    label = "davies"
    expt_params_normal_sd = 0.11


protocol = 1

parallel = True

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)
# defined in generate_synthetic_data.py, should abstract(?) this so it's definitely consistent
top_theta = original_gs
top_sigma_squared = (expt_params_normal_sd * original_gs)**2

g_labels = ["${}$".format(g) for g in g_parameters]

norm_pdf = st.norm.pdf
sqrt = np.sqrt

num_pts = 501

which_g = 0

fig, (ax,ax2) = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
ax.grid()
ax2.grid()

x_range = np.linspace(0.5*original_gs[which_g],1.5*original_gs[which_g],num_pts)
true_pdf = norm_pdf(x_range,top_theta[which_g],np.sqrt(top_sigma_squared[which_g])) # not sure if there should be a square
ax.plot(x_range,true_pdf,label='True',color=colors[-1],lw=3)
ax2.plot(x_range,true_pdf,label='True',color=colors[-1],lw=3)
ax.set_title('Posterior predictive')
ax2.set_title('Maximum likelihood')
how_many_traces = [2,4,8,16,32]
for j, N_e in enumerate(how_many_traces):
    mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
    print "\n", mcmc_file, "\n"
    chain = np.loadtxt(mcmc_file,usecols=range(which_g, (2+N_e)*num_gs, num_gs))
    print chain[0,:]
    print chain[1,:]
    saved_its, _ = chain.shape
    burn = saved_its/2
    length = saved_its - burn
    chain = chain[burn:, :]

    sum_pdf = np.zeros(len(x_range))


    for i in xrange(length):
        temp_pdf = norm_pdf(x_range,chain[i,0],sqrt(chain[i,1]))
        sum_pdf += temp_pdf/length
    #sum_pdf /= length

    ax.plot(x_range,sum_pdf,label="$N_e = {}$".format(N_e),color=colors[j],lw=3)
    
    expt_chains = chain[:, 2:]
    expt_means = np.mean(expt_chains, axis=0)
    loc1, scale1 = st.norm.fit(expt_means)
    ax2.plot(x_range, norm_pdf(x_range, loc1, scale1), color=colors[j], lw=3,label="$N_e = {}$".format(N_e))
    
ax.set_xlabel(g_labels[which_g])
ax2.set_xlabel(g_labels[which_g])
ax.set_ylabel("Probability density")
ax.legend(loc=2, fontsize=12)
ax2.legend(loc=2, fontsize=12)

#for i in xrange(max(how_many_traces)):
#    ax.axvline(true_params[i, which_g], color='blue')
fig.tight_layout()
plt.show(block=True)






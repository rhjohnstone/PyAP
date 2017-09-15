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


colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']


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
N_e = args.num_traces

parallel = True




original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)
# defined in generate_synthetic_data.py, should abstract(?) this so it's definitely consistent
scale_for_generating_expt_params = 0.1
top_theta = original_gs
top_sigma_squared = (scale_for_generating_expt_params * original_gs)**2

norm_pdf = st.norm.pdf

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ax.grid()
num_pts = 501
x_range = np.linspace(0.5*original_gs[0],1.5*original_gs[0],num_pts)
true_pdf = st.norm.pdf(x_range,top_theta[0],np.sqrt(top_sigma_squared[0])) # not sure if there should be a square
ax.plot(x_range,true_pdf,label='True',color=colors[0],lw=3)
for N_e in [2,4,8,16]:
    mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
    chain = np.loadtxt(mcmc_file,usecols=range(num_gs))
    saved_its, _ = chain.shape
    burn = saved_its/4
    length = saved_its - burn

    sum_pdf = np.zeros(len(x_range))


    for i in xrange(burn, saved_its):
        temp_pdf = norm_pdf(x_range,chain[i,0],np.sqrt(chain[i,1]))
        sum_pdf += temp_pdf
    sum_pdf /= length

    ax.plot(x_range,sum_pdf,label="$N_e = {}$".format(N_e),color=colors[1],lw=3)
ax.legend()
plt.show(block=True)
sys.exit()

N_e = 5
directory = "../output/hierarchical/synthetic/m"+str(model)+"/p"+str(protocol)+"/post_cmaes/"+str(N_e)+"_expts/"
if not os.path.exists(directory):
    os.makedirs(directory)
chain_dir = directory + "chain/"
if not os.path.exists(chain_dir):
    os.makedirs(chain_dir)

chain_segs = os.listdir(chain_dir)
chain_segs.sort()

total_length = 0
norm_pdf = st.norm.pdf
for j,just_file in enumerate(chain_segs):
    output_file = chain_dir+just_file
    MCMC = np.loadtxt(output_file,usecols=[0,num_params])
    end = MCMC.shape[0]
    if j==0:
        burn = end/2
    else:
        burn = 0
    length = end-burn
    total_length += length
    for i in range(length):
        temp_pdf = norm_pdf(x_range,MCMC[burn+i,0],np.sqrt(MCMC[burn+i,1]))
        sum_pdf10 += temp_pdf
sum_pdf10 /= total_length

ax.plot(x_range,sum_pdf10,label="$N_e = {}$".format(N_e),color=colors[2],lw=3)

N_e = 10
directory = "../output/hierarchical/synthetic/m"+str(model)+"/p"+str(protocol)+"/post_cmaes/"+str(N_e)+"_expts/"
if not os.path.exists(directory):
    os.makedirs(directory)
chain_dir = directory + "chain/"
if not os.path.exists(chain_dir):
    os.makedirs(chain_dir)

chain_segs = os.listdir(chain_dir)
chain_segs.sort()

total_length = 0
norm_pdf = st.norm.pdf
for j,just_file in enumerate(chain_segs):
    output_file = chain_dir+just_file
    MCMC = np.loadtxt(output_file,usecols=[0,num_params])
    end = MCMC.shape[0]
    if j==0:
        burn = end/2
    else:
        burn = 0
    length = end-burn
    total_length += length
    for i in range(length):
        temp_pdf = norm_pdf(x_range,MCMC[burn+i,0],np.sqrt(MCMC[burn+i,1]))
        sum_pdf10 += temp_pdf
sum_pdf10 /= total_length

ax.plot(x_range,sum_pdf10,label="$N_e = {}$".format(N_e),color=colors[3],lw=3)




ax.set_ylabel('Probability density')
ax.set_xlabel(r'$G_{Na}$')
ax.set_xlim(2000,5500)
ax.set_ylim(0,0.008)
plt.legend()
fig.tight_layout()
fig.savefig(directory+str(N_e)+'_G_Na_expts_inferred_and_true.eps')
plt.close()

print "\nDone!\n"

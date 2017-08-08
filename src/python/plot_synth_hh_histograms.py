import pyap_setup as ps
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse

# Already discarded burn-in when saving MCMC file, so no need to discard here

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
requiredNamed.add_argument("--seed", type=int, help="python random seed for initial position", required=True)
parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
parser.add_argument("--non-adaptive", action="store_true", help="do not adapt proposal covariance matrix", default=False)
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

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)

labels = g_parameters+[r"\sigma"]

mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name, args.unscaled, args.non_adaptive)
spl = mcmc_file.split('.')
mcmc_file = "{}_seed_{}.{}".format(spl[0], args.seed, spl[1])
spl = log_file.split('.')
log_file = "{}_seed_{}.{}".format(spl[0], args.seed, spl[1])
try:
    chain = np.loadtxt(mcmc_file)
except:
    sys.exit("\nCan't find (or load) {}\n".format(mcmc_file))
    
saved_its, num_params_plus_1 = chain.shape
burn = saved_its/4

for i in xrange(num_gs+1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title("{} - seed {}".format(model_name, args.seed))
    ax.set_ylabel('Marginal density')
    if i < num_gs:
        ax.set_xlabel("$"+g_parameters[i]+"$")
        savelabel = png_dir+g_parameters[i]+'_marginal.png'
    else:
        ax.set_xlabel(r"$\sigma$")
        savelabel = png_dir+'sigma_marginal.png'
    ax.hist(chain[burn:,i], normed=True, bins=40, color='blue', edgecolor='blue')
    fig.tight_layout()
    fig.savefig(savelabel)
    plt.close()

    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_title("{} - seed {}".format(model_name, args.seed))
ax.plot(chain[burn:,-1], lw=1, color='blue')
ax.set_xlabel("Saved iteration")
ax.set_ylabel('Log-target')
fig.tight_layout()
fig.savefig(png_dir+'log_target.png')
plt.close()

# plot scatterplot matrix of posterior(s)
colormin, colormax = 1e9,0
norm = matplotlib.colors.Normalize(vmin=5,vmax=10)
hidden_labels = []
count = 0
# there's probably a better way to do this
# I plot all the histograms to normalize the colours, in an attempt to give a better comparison between the pairwise plots
num_params = num_gs+1
while count < 2:
    axes = {}
    matrix_fig = plt.figure(figsize=(3*num_params,3*num_params))
    for i in range(num_params):
        for j in range(i+1):
            ij = str(i)+str(j)
            subplot_position = num_params*i+j+1
            if i==j:
                axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position)
                axes[ij].hist(chain[burn:,i],bins=50,normed=True,color='blue', edgecolor='blue')
            elif j==0: # this column shares x-axis with top-left
                axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes["00"])
                counts, xedges, yedges, Image = axes[ij].hist2d(chain[burn:,j],chain[burn:,i],cmap='hot_r',bins=50,norm=norm)
                maxcounts = np.amax(counts)
                if maxcounts > colormax:
                    colormax = maxcounts
                mincounts = np.amin(counts)
                if mincounts < colormin:
                    colormin = mincounts
            else:
                axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes[str(j)+str(j)],sharey=axes[str(i)+"0"])
                counts, xedges, yedges, Image = axes[ij].hist2d(chain[burn:,j],chain[burn:,i],cmap='hot_r',bins=50,norm=norm)
                maxcounts = np.amax(counts)
                if maxcounts > colormax:
                    colormax = maxcounts
                mincounts = np.amin(counts)
                if mincounts < colormin:
                    colormin = mincounts
            axes[ij].xaxis.grid()
            if (i!=j):
                axes[ij].yaxis.grid()
            if i!=num_params-1:
                hidden_labels.append(axes[ij].get_xticklabels())
            if j!=0:
                hidden_labels.append(axes[ij].get_yticklabels())
            if i==j==0:
                hidden_labels.append(axes[ij].get_yticklabels())
            if i==num_params-1:
                axes[str(i)+str(j)].set_xlabel("$"+labels[j]+"$")
            if j==0 and i>0:
                axes[str(i)+str(j)].set_ylabel("$"+labels[i]+"$")
                
            plt.xticks(rotation=30)
    norm = matplotlib.colors.Normalize(vmin=colormin,vmax=colormax)
    count += 1

    
plt.setp(hidden_labels, visible=False)

matrix_fig.tight_layout()
matrix_fig.savefig(png_dir+'scatterplot_matrix.png')
#matrix_fig.savefig(images_dir+"{}_{}_scatterplot_matrix.pdf".format(drug,channel))
plt.close()


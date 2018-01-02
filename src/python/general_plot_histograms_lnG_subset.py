import pyap_setup as ps
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
from scipy.stats import norm
#import ap_simulator


"""def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    try:
        return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    except ap_simulator.CPPException, e:
        print e.GetShortMessage
        print "temp_g_params:\n", temp_g_params
        print "original_gs:\n", original_gs
        return np.zeros(len(expt_times))"""


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
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

trace_number = int(trace_name.split("_")[-2])  # just for Roche

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

if pyap_options["model_number"]==3:  # LR
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] * 1e-6
elif pyap_options["model_number"]==4:  # TT
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
    indices_to_keep = [0, 1, 2, 3, 4, 6]  # which currents we are fitting
elif pyap_options["model_number"]==5:  # OH
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
elif pyap_options["model_number"]==7:  # Pa
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-12
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-12
    indices_to_keep = [0, 1, 2, 3, 4, 9, 11]  # which currents we are fitting
num_params_to_fit = len(indices_to_keep) + 1  # +1 for sigma



original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)


labels = g_parameters+[r"\sigma"]

mcmc_file, log_file, png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name)
try:
    chain = np.loadtxt(mcmc_file)
    if expt_name=="roche_ten_tusscher_correct_units" or expt_name=="roche_paci_correct_units":
        saved_its = chain.shape[0]
        chain = chain[saved_its/2:, :]
except:
    sys.exit("\nCan't find (or load) {}\n".format(mcmc_file))
    

if expt_name=="roche_ten_tusscher":
    saved_its = chain.shape[0]
    sl_chain = chain[saved_its/4:, :]
    
saved_its, num_params_plus_1 = chain.shape

best_all = chain[np.argmax(chain[:,-1]),:]
best_params = best_all[:-1]
best_ll = best_all[-1]
best_fit_gs = best_params[:-1]

"""figg = plt.figure(figsize=(4,4))
axx = figg.add_subplot(111)
axx.grid()
axx.plot(expt_times, expt_trace, color='red', label='Expt')
axx.plot(expt_times, solve_for_voltage_trace(best_fit_gs, ap_model), color='blue', label='Best MCMC fit')
axx.set_xlabel('Time (ms)')
axx.set_ylabel('Membrane voltage (mV)')
figg.tight_layout()
figg.savefig(png_dir+"best_mcmc_fit.png")
figg.savefig(png_dir+"best_mcmc_fit.pdf")"""

m_true = np.log(original_gs)
sigma2_true = 0.01

prior_mean = np.log(original_gs)
prior_sd = 0.5*np.log(10)  # s.d. of Normal priors on lnGs

cs = ['#1b9e77','#d95f02','#7570b3','#e7298a']
num_prior_pts = 201
num_ticks = 6
for i in xrange(num_params_to_fit):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_ylabel('Normalised frequency')
    if i < num_params_to_fit-1:
        ax.set_xlabel("log(${}$)".format(g_parameters[indices_to_keep[i]]))
        savelabel = png_dir+g_parameters[indices_to_keep[i]]+'_marginal.png'
    else:
        ax.set_xlabel(r"$\sigma$")
        savelabel = png_dir+'sigma_marginal.png'
    ax.hist(chain[:,i], normed=True, bins=40, color=cs[0], edgecolor=cs[0])
    ax2 = ax.twinx()
    ax2.grid()
    xlim = ax.get_xlim()
    xlength = xlim[1]-xlim[0]
    x = np.linspace(xlim[0]-0.2*xlength, xlim[1]+0.2*xlength, num_prior_pts)
    ax.set_xlim(x[0], x[-1])
    if i < num_params_to_fit-1:
        ax2.plot(x, norm.pdf(x, loc=prior_mean[indices_to_keep[i]], scale=prior_sd), lw=2, color=cs[1], label='Prior')
        ax2.axvline(best_params[i], color=cs[2], lw=2, label="Max PD")
    else:
        ax2.plot(0.5, 0, 'x', color=cs[3], ms=10, mew=2, label='Expt', clip_on=False, zorder=10)
        if (x[0] > 1e-3) and (x[-1] < 25):
            ax2.axhline(1./(25.-1e-3), lw=2, color=cs[1], label='Prior')
    
    ax.set_xticks(np.round(np.linspace(x[0], x[-1], 6), 3))
    ax2.set_ylim(0, ax2.get_ylim()[1])
    ax2.set_ylabel('Probability density')
    ax2.legend(loc='best', fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yticks(np.round(np.linspace(0., ax.get_ylim()[1], num_ticks), 3))
    ax2.set_yticks(np.round(np.linspace(0., ax2.get_ylim()[1], num_ticks),3))
    fig.tight_layout()
    fig.savefig(savelabel)
    plt.close()

    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(chain[:,-1], lw=1, color='blue')
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
    matrix_fig = plt.figure(figsize=(2*num_params_to_fit,2*num_params_to_fit))
    for i in range(num_params_to_fit):
        for j in range(i+1):
            ij = str(i)+str(j)
            subplot_position = num_params*i+j+1
            if i==j:
                axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position)
                axes[ij].hist(chain[:,i],bins=50,normed=True,color='blue', edgecolor='blue')
            elif j==0: # this column shares x-axis with top-left
                axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes["00"])
                counts, xedges, yedges, Image = axes[ij].hist2d(chain[:,j],chain[:,i],cmap='hot_r',bins=50,norm=norm)
                maxcounts = np.amax(counts)
                if maxcounts > colormax:
                    colormax = maxcounts
                mincounts = np.amin(counts)
                if mincounts < colormin:
                    colormin = mincounts
            else:
                axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes[str(j)+str(j)],sharey=axes[str(i)+"0"])
                counts, xedges, yedges, Image = axes[ij].hist2d(chain[:,j],chain[:,i],cmap='hot_r',bins=50,norm=norm)
                maxcounts = np.amax(counts)
                if maxcounts > colormax:
                    colormax = maxcounts
                mincounts = np.amin(counts)
                if mincounts < colormin:
                    colormin = mincounts
            axes[ij].xaxis.grid()
            axes[ij].tick_params(axis='both', which='major', labelsize=10)
            axes[ij].tick_params(axis='both', which='minor', labelsize=8)
            num_ticks = 3
            start, end = axes[ij].get_xlim()
            axes[ij].xaxis.set_ticks(np.linspace(start, end, num_ticks+2)[1:-1])
            start, end = axes[ij].get_ylim()
            axes[ij].yaxis.set_ticks(np.linspace(start, end, num_ticks+2)[1:-1])
            if (i!=j):
                axes[ij].yaxis.grid()
            if i!=num_params-1:
                hidden_labels.append(axes[ij].get_xticklabels())
            if j!=0:
                hidden_labels.append(axes[ij].get_yticklabels())
            if i==j==0:
                hidden_labels.append(axes[ij].get_yticklabels())
            if i==num_params-1:
                if j==num_params-1:
                    axes[str(i)+str(j)].set_xlabel("$"+labels[indices_to_keep[j]]+"$")
                else:
                    axes[str(i)+str(j)].set_xlabel("log($"+labels[indices_to_keep[j]]+"$)")
            if j==0 and i>0:
                if i==num_params_to_fit-1:
                    axes[str(i)+str(j)].set_ylabel("$"+labels[indices_to_keep[i]]+"$")
                else:
                    axes[str(i)+str(j)].set_ylabel("log($"+labels[indices_to_keep[i]]+"$)")
                
            plt.xticks(rotation=30)
    norm = matplotlib.colors.Normalize(vmin=colormin,vmax=colormax)
    count += 1

    
plt.setp(hidden_labels, visible=False)

matrix_fig.tight_layout()
matrixpng = png_dir+'scatterplot_matrix.png'
print matrixpng
matrix_fig.savefig(matrixpng)
plt.close()


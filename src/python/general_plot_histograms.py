import pyap_setup as ps
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

original_gs, g_parameters = ps.get_original_params(ps.pyap_options["model_number"])
num_gs = len(original_gs)

mcmc_file, log_file, png_dir = ps.mcmc_exp_scaled_file_log_file_and_figs_dirs(ps.pyap_options["model_number"])
try:
    chain = np.loadtxt(mcmc_file)
except:
    sys.exit("\nCan't find (or load) MCMC output file\n")

for i in xrange(num_gs+1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_ylabel('Marginal density')
    if i < num_gs:
        ax.set_xlabel("$"+g_parameters[i]+"$")
        savelabel = png_dir+g_parameters[i]+'_marginal.png'
    else:
        ax.set_xlabel(r"$\sigma$")
        savelabel = png_dir+'sigma_marginal.png'
    ax.hist(chain[:,i], normed=True, bins=40, color='blue', edgecolor='blue')
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


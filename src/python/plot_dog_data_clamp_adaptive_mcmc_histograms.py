import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyap_setup as ps

exp_scaling = True

model_number = 9
trace_number = 150

original_gs, g_parameters = ps.get_original_params(model_number)
num_gs = len(original_gs)

if exp_scaling:
    mcmc_dir, mcmc_file = ps.dog_data_clamp_exp_scaled_mcmc_file(model_number, trace_number)
else:
    mcmc_dir, mcmc_file = ps.dog_data_clamp_unscaled_mcmc_file(model_number, trace_number)
chain = np.loadtxt(mcmc_file)

for i in xrange(num_gs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.hist(chain[:,i], normed=True, bins=40, color='blue', edgecolor='blue')
    ax.set_xlabel(g_parameters[i])
    ax.set_ylabel('Marginal density')
    fig.tight_layout()
    fig.savefig(mcmc_dir+g_parameters[i]+'_marginal.png')
    plt.close()
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(chain[:,-1], lw=2, color='blue')
ax.set_xlabel("Saved iteration")
ax.set_ylabel('Log-target')
fig.tight_layout()
fig.savefig(mcmc_dir+'log_target.png')
plt.close()

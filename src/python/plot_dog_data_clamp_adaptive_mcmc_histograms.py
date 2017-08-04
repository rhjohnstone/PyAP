import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyap_setup as ps

model_number = 9
#trace_number = 150

def plot_all(trace_number, exp_scaling):
    original_gs, g_parameters, model_name = ps.get_original_params(model_number)
    num_gs = len(original_gs)

    if exp_scaling:
        mcmc_dir, mcmc_file = ps.dog_data_clamp_exp_scaled_mcmc_file(model_number, trace_number)
    else:
        mcmc_dir, mcmc_file = ps.dog_data_clamp_unscaled_mcmc_file(model_number, trace_number)
    chain = np.loadtxt(mcmc_file)

    for i in xrange(num_gs+1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        #if exp_scaling:
        #    ax.set_title("Re-scaled, but MCMC done with exp scaling")
        #    params_to_hist = original_gs[i]**chain[:,i]
        #else:
        #    ax.set_title("MCMC done on unscaled params")
        ax.set_ylabel('Marginal density')
        if i < num_gs:
            ax.set_xlabel("$"+g_parameters[i]+"$")
            savelabel = mcmc_dir+g_parameters[i]+'_marginal.png'
        else:
            ax.set_xlabel(r"$\sigma$")
            savelabel = mcmc_dir+'sigma_marginal.png'
        ax.hist(chain[:,i], normed=True, bins=40, color='blue', edgecolor='blue')
        fig.tight_layout()
        fig.savefig(savelabel)
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
    return None
    
for es in [True, False]:
    for trace_number in xrange(150,150+16):
        try:
            plot_all(trace_number, es)
        except:
            print "Can't find trace", trace_number


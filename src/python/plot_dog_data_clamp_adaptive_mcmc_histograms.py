import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyap_setup as ps


def exponential_scaling(unscaled_params):
    return original_gs ** unscaled_params


exp_scaling = True

model_number = 9
#trace_number = 150

def plot_all(trace_number):
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
        if exp_scaling:
            ax.set_title("Re-scaled, but MCMC done with exp scaling")
            params_to_hist = original_gs[i]**chain[:,i]
        else:
            ax.set_title("MCMC done on unscaled params")
            params_to_hist = chain[:,i]
        ax.hist(params_to_hist, normed=True, bins=40, color='blue', edgecolor='blue')
        ax.set_xlabel("$"+g_parameters[i]+"$")
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
    return None
    
for trace_number in xrange(150,150+16):
    try:
        plot_all(trace_number)
    except:
        print "Can't find trace", trace_number


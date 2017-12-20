import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import invgamma, norm

start = time.time()

inf = np.inf
dot = np.dot
multivariate_normal = npr.multivariate_normal
npcopy = np.copy
exp = np.exp

python_seed = 1
npr.seed(python_seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("-T", "--num-samples", type=int, help="number of samples to plot priors for", required=True)
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

T = args.num_samples

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
    
model_number = pyap_options["model_number"]

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
num_gs = len(original_gs)


split_trace_name = trace_name.split("_")
first_trace_number = int(split_trace_name[-1])  # need a specific-ish format currently

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)


parallel = True



m_true = np.log(original_gs)
sigma2_true = 0.01


mu = m_true
alpha = 4.*np.ones(num_gs)
beta = (alpha+1.) * 0.04
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

old_eta_js = np.vstack((mu, nu, alpha, beta)).T

print "old_eta_js:\n", old_eta_js


#sys.exit()

uniform_noise_prior = [1e-3,25.]

def new_eta(old_eta, samples): # for sampling from conjugate prior-ed N-IG
    x_bar = np.mean(samples)
    mu, nu, alpha, beta = 1.*old_eta
    new_mu = ((nu*mu + N_e*x_bar) / (nu + N_e))
    new_nu = nu + N_e
    new_alpha = alpha + 0.5*N_e
    new_beta = beta + 0.5*np.sum((samples-x_bar)**2) + 0.5*((N_e*nu)/(nu+N_e))*(x_bar-mu)**2
    return new_mu, new_nu, new_alpha, new_beta
    
randn = npr.randn
sqrt = np.sqrt
def sample_from_N_IG(eta):
    mu, nu, alpha, beta = eta
    sigma_squared_sample = invgamma.rvs(alpha,scale=beta)
    sample = mu + sqrt(sigma_squared_sample/nu)*randn()
    return sample, sigma_squared_sample
    
    
def just_sample_m(Mu, Lambda, Sigma2):
    return norm.rvs(loc=Mu, scale=sqrt(Sigma2/Lambda))
    



num_pts = 101
#p = 0
x1 = np.linspace(4, 5, num_pts)
x2 = np.linspace(0.005, 0.05, num_pts)
xs = [x1, x2]



cs = ['#1b9e77','#d95f02','#7570b3']

nums_expts = [2, 4]#, 8, 16, 32]

total_nums_expts = len(nums_expts)
color_idx = np.linspace(0, 1, total_nums_expts)
labels = ("Prior",) + tuple(["$N_e = {}$".format(n) for n in nums_expts])

colors = [plt.cm.winter(color_idx[a]) for a in xrange(total_nums_expts)]

lines = []
p_s = [0, 4, 11]


fig, axs = plt.subplots(len(p_s), 2, figsize=(7,3*len(p_s)))

for p in p_s:
    a, b = alpha[p], beta[p]
    s2_prior = invgamma.pdf(x2, a, loc=0, scale=b)

    m_prior = np.zeros(num_pts)
    for t in xrange(T):
        s2_sample = invgamma.rvs(a, loc=0, scale=b)
        m_prior += norm.pdf(x1, loc=mu[p], scale=sqrt(s2_sample/nu[p]))
    m_prior /= T


    xlabels = ["$m$", "$s^2$"]
    priors = [m_prior, s2_prior]
    for j in xrange(2):
        line, = axs[p][j].plot(xs[j], priors[j], color=cs[1], lw=2, zorder=0)
        axs[p][j].grid()
        axs[p][j].set_xlabel(xlabels[j] + " $({})$".format(g_parameters[p]), fontsize=16)
        axs[p][j].set_ylabel("Probability density")
        axs[p][j].set_xlim(xs[j][0], xs[j][-1])
        
    if p==0:
        lines.append(line)
    
    
    for a, N_e in enumerate(nums_expts):
        hmcmc_file, log_file, h_png_dir, pdf_dir = ps.hierarchical_lnG_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)
        h_chain = np.loadtxt(hmcmc_file,usecols=[p, num_gs+p])
        saved_its = h_chain.shape[0]

        for j in xrange(2):
            axs[p][j].hist(h_chain[:, j], normed=True, bins=40, lw=0, color=colors[a], alpha=1.5/len(nums_expts), zorder=10)
    
lines += [mpatches.Patch(color=color) for color in colors]

leg = fig.legend(lines, labels, loc="upper center", ncol=1+len(nums_expts)/2, bbox_to_anchor=(0.5, 1.05))

fig.tight_layout()

fig_file = h_png_dir + "{}_m_s2_priors_marginals_{}.png".format(expt_name, max(nums_expts))
print fig_file

#fig.savefig(fig_file, bbox_extra_artists=(leg,), bbox_inches='tight', pad_inches=0.05)

plt.show()
#for t in xrange(T):
    



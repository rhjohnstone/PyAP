import ap_simulator
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm

start = time.time()

inf = np.inf
dot = np.dot
multivariate_normal = npr.multivariate_normal
npcopy = np.copy
exp = np.exp
sqrt = np.sqrt
npsum = np.sum
nplog = np.log
nprrandn = npr.randn

def solve_for_voltage_trace(temp_g_params, ap_model_index):
    if np.any(temp_g_params<0):
        return np.zeros(num_pts)
    ap_models[ap_model_index].SetToModelInitialConditions()
    try:
        return ap_models[ap_model_index].SolveForVoltageTraceWithParams(temp_g_params)
    except:
        print "Failed to solve"
        print "temp_g_params:", temp_g_params
        return np.zeros(num_pts)
        

def solve_star(temp_g_params_and_ap_model_index):
    return solve_for_voltage_trace(*temp_g_params_and_ap_model_index)
    

python_seed = 1
npr.seed(python_seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("--num-traces", type=int, help="number of traces to fit to, including the one specified as argument", required=True)
requiredNamed.add_argument("-i", "--iterations", type=int, help="total MCMC iterations", required=True)
parser.add_argument("-nc", "--num-cores", type=int, help="number of cores to parallelise solving expt traces", default=1)
parser.add_argument("--cheat", action="store_true", help="for synthetic data: start MCMC from parameter values used to generate data", default=False)
parser.add_argument("--different", action="store_true", help="use different initial guess for some params", default=False)
#parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
args, unknown = parser.parse_known_args()

N_e = args.num_traces

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

#num_processors = multiprocessing.cpu_count()
#num_processes = min(num_processors-1,N_e) # counts available cores and makes one fewer process


split_trace_name = trace_name.split("_")
first_trace_number = int(split_trace_name[-1])  # need a specific-ish format currently
trace_numbers = range(first_trace_number, first_trace_number+N_e)

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)


best_fits_params = np.zeros((N_e, num_gs))
expt_traces = []
ap_models = []
temp_test_traces_cur = []
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.grid()
for i, t in enumerate(trace_numbers):
    if (0 <= first_trace_number <= 9):
        temp_trace_path = "{}_{}.csv".format(trace_path[:-6], t)
    elif (10 <= first_trace_number <= 99):
        temp_trace_path = "{}_{}.csv".format(trace_path[:-7], t)
    elif (100 <= first_trace_number <= 999):
        temp_trace_path = "{}_{}.csv".format(trace_path[:-8], t)
    temp_times, temp_trace = np.loadtxt(temp_trace_path,delimiter=',').T
    if i==0:
        expt_times = temp_times
    expt_traces.append(npcopy(temp_trace))
    #ax.plot(expt_times, expt_traces[i])
    if not args.cheat:
        temp_trace_name = trace_name[:-3]+str(t)
        cmaes_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(pyap_options["model_number"], expt_name, temp_trace_name, unscaled=False)
        all_best_fits = np.loadtxt(cmaes_file)
        best_index = np.argmin(all_best_fits[:, -1])
        best_params = all_best_fits[best_index, :-1]
    else:
        best_params = np.loadtxt('/'.join( split_trace_path[:5] ) + "/expt_params.txt")[t, :]
        if args.different: # 9,10,11
            for j in [9,10,11]:
                best_params[j] = 10 * original_gs[j] * npr.rand()
    best_fits_params[i, :] = npcopy(best_params)
    temp_ap_model = ap_simulator.APSimulator()
    if (data_clamp_on < data_clamp_off):
        temp_ap_model.DefineStimulus(0, 1, 1000, 0)  # no injected stimulus current
        temp_ap_model.DefineModel(pyap_options["model_number"])
        temp_ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
        temp_ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, npcopy(temp_trace))
    else:
        temp_ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
        temp_ap_model.DefineModel(pyap_options["model_number"])
    temp_ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
    temp_ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
    temp_ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
    temp_ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
    temp_ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
    temp_ap_model.SetNumberOfSolves(pyap_options["num_solves"])
    ap_models.append(temp_ap_model)
    temp_test_traces_cur.append(npcopy(solve_for_voltage_trace(best_params, i)))
expt_traces = np.array(expt_traces)
temp_test_traces_cur = np.array(temp_test_traces_cur)


starting_points = npcopy(best_fits_params)


starting_mean = np.mean(starting_points,axis=0)


parallel = True

mcmc_file, log_file, png_dir, pdf_dir = ps.hierarchical_mcmc_files(pyap_options["model_number"], expt_name, trace_name, N_e, parallel)

initial_it_file = log_file[:-4]+"_initial_iteration.txt"

coeffs = [1, -1, 0, 0, -16]
roots = np.roots(coeffs)
assert(np.isreal(roots[0]))
v = float(roots[0])  # manually ascertained

gamma_hyperparams = np.zeros((num_gs,2))
gamma_hyperparams[:,0] = 3.  # alpha
gamma_hyperparams[:,1] = (gamma_hyperparams[:,0]-1.)*np.log(v)  # beta

normal_hyperparams = np.zeros((num_gs,2))
normal_hyperparams[:,0] = np.log(original_gs) + gamma_hyperparams[:,1] / (gamma_hyperparams[:,0]-1.)  # s
normal_hyperparams[:,1] = 4.  # lambda


def update_normal_hyperparams(old_hypers, nu_hat, samples):
    s, t = old_hypers
    new_s = (nu_hat * npsum(np.log(samples)) + t*s) / (N_e * nu_hat + t)
    new_t = N_e * nu_hat + t
    return new_s, new_t


def sample_from_updated_top_mu_normal(new_s, new_t):
    mu_sample = norm.rvs(loc=new_s, scale=sqrt(new_t))
    return mu_sample


def update_gamma_hyperparams(old_hypers, g_hat, samples):
    alpha, beta = old_hypers
    new_alpha = alpha + N_e/2.
    new_beta = beta + 0.5*npsum((np.log(samples)-g_hat)**2)
    return new_alpha, new_beta


def sample_from_updated_top_tau_gamma(new_alpha, new_beta):
    tau_sample = gamma.rvs(new_alpha, scale=1./new_beta)
    return tau_sample


num_pts = len(expt_times)

#sys.exit()

uniform_noise_prior = [0.,25.]
    
    
def log_pi_g_i(g_i, mus, taus, sigma, data_i, test_i):
    if np.any(g_i < 0) or (not (uniform_noise_prior[0] < sigma < uniform_noise_prior[1])):
        return -inf
    else:
        return -npsum((data_i - test_i)**2)/(2.*sigma**2) - npsum(nplog(g_i)) - npsum(taus*(nplog(g_i)-mus)**2 / 2.)


def log_pi_sigma(expt_datas, test_datas, sigma):
    if (not (uniform_noise_prior[0] < sigma < uniform_noise_prior[1])):
        return -inf
    else:
        return -N_e*num_pts*np.log(sigma) - npsum((expt_datas-test_datas)**2) / (2*sigma**2)


def compute_initial_sigma(expt_datas, test_datas):
    return sqrt(npsum((expt_datas-test_datas)**2) / (N_e*num_pts))


s = 0.2
mu = np.log(original_gs) - s**2/2
tau = np.ones(num_gs)/np.sqrt(s)


mus_cur = npcopy(mu)
taus_cur = npcopy(tau)
g_is_cur = npcopy(starting_points)

taus_cur[taus_cur<=0] = 1e-3
g_is_cur[g_is_cur<=0] = 1e-3

noise_sigma_cur = 0.5

cov_proposal_scale = 0.0001
sigma_proposal_scale = 1.


print "\n\n", g_is_cur, "\n\n"


from multiprocessing import Pool
global noise_sigma_cur

print "\nPARALLEL\n"

thinning = 5
MCMC_iterations = args.iterations
num_saved_its = MCMC_iterations / thinning + 1
burn = num_saved_its / 4
when_to_adapt = 100*num_gs

status_when = MCMC_iterations / 100

print "mus_cur:", mus_cur
print "noise_sigma_cur:", noise_sigma_cur

MCMC = np.zeros((num_saved_its, (2+N_e)*num_gs+1))
MCMC[0, :] = np.concatenate((mus_cur, taus_cur, g_is_cur.flatten(), [noise_sigma_cur]))
np.savetxt(initial_it_file, MCMC[0, :])

covariances = []
for i in range(N_e):
    covariances.append(cov_proposal_scale*np.diag(g_is_cur[i,:]))
print "covariances:\n", covariances, "\n"

means = npcopy(g_is_cur)
print "means:\n", means, "\n"

logas = [0.]*N_e
acceptances = [0.]*N_e
sigma_loga = 0.
sigma_acceptance = 0.

#if t > 1000*number_of_parameters:
def update_covariance_matrix(t, thetaCur, mean_estimate, cov_estimate, loga, accepted):
    s = t - when_to_adapt
    gamma_s = 1/(s+1)**0.6
    temp_covariance_bit = np.array([thetaCur-mean_estimate])
    new_cov_estimate = (1-gamma_s) * cov_estimate + gamma_s * dot(np.transpose(temp_covariance_bit),temp_covariance_bit)
    new_mean_estimate = (1-gamma_s) * mean_estimate + gamma_s * thetaCur
    new_loga = loga + gamma_s*(accepted-0.25)
    return new_cov_estimate, new_mean_estimate, new_loga
    

adapt_started = True
g_is_stars = np.zeros((N_e, num_gs))

pool = Pool(args.num_cores)
t = 1
print "About to start MCMC\n"
while (t <= MCMC_iterations):
    for j in xrange(num_gs):
        # sample from g_hat conditional
        new_s, new_lamb = update_normal_hyperparams(normal_hyperparams[j,:], taus_cur[j], g_is_cur[:,j])
        temp_mu_cur = sample_from_updated_top_mu_normal(new_s, new_lamb)
        mus_cur[j] = temp_mu_cur
        # sample from nu_hat conditional
        new_alpha, new_beta = update_gamma_hyperparams(gamma_hyperparams[j,:], temp_mu_cur, g_is_cur[:,j])
        temp_tau_cur = sample_from_updated_top_tau_gamma(new_alpha, new_beta)  # already positive
        taus_cur[j] = temp_tau_cur
                

    # theta i's for each experiment
    
    for i in xrange(N_e):
        g_is_stars[i, :] = multivariate_normal(g_is_cur[i, :], exp(logas[i])*covariances[i])
        
                
    g_is_stars_and_ap_model_index = zip(g_is_stars, range(N_e))
    
    temp_test_traces_star = pool.map_async(solve_star, g_is_stars_and_ap_model_index).get(999)


    for i in xrange(N_e):
        temp_test_trace_star = temp_test_traces_star[i]
    
        target_cur = log_pi_g_i(g_is_cur[i, :], mus_cur, taus_cur, noise_sigma_cur, expt_traces[i], temp_test_traces_cur[i])
        target_star = log_pi_g_i(g_is_stars[i, :], mus_cur, taus_cur, noise_sigma_cur, expt_traces[i], temp_test_trace_star)
        u = npr.rand()
        if (np.log(u) < target_star - target_cur):
            g_is_cur[i, :] = npcopy(g_is_stars[i, :])
            temp_test_traces_cur[i] = npcopy(temp_test_trace_star)
            accepted = 1
        else:
            accepted = 0
        if (t > when_to_adapt):
            temp_cov, temp_mean, temp_loga = update_covariance_matrix(t, g_is_cur[i, :], means[i], covariances[i], logas[i], accepted)
            covariances[i] = npcopy(temp_cov)
            means[i] = npcopy(temp_mean)
            logas[i] = temp_loga
        acceptances[i] = (t*acceptances[i] + accepted)/(t+1.)
    # noise sigma
    noise_sigma_star = noise_sigma_cur + exp(sigma_loga)*sigma_proposal_scale*nprrandn()
    sigma_target_star = log_pi_sigma(expt_traces, temp_test_traces_cur, noise_sigma_star)
    sigma_target_cur = log_pi_sigma(expt_traces, temp_test_traces_cur, noise_sigma_cur)
    u = npr.rand()
    if (np.log(u) < sigma_target_star - sigma_target_cur):
        noise_sigma_cur = noise_sigma_star
        accepted = 1
    else:
        accepted = 0
    sigma_acceptance = (t*sigma_acceptance + accepted)/(t+1)
    if (t > when_to_adapt):
        r = t - when_to_adapt
        gamma_r = 1/(r+1)**0.6
        sigma_loga += gamma_r*(accepted-0.25)
    if ( t%thinning == 0 ):   
        MCMC[t/thinning, :] = np.concatenate((mus_cur,taus_cur,g_is_cur.flatten(),[noise_sigma_cur]))
    t += 1
    if ( t%status_when==0 ):
        print t, "iterations"
        print "logas =", logas
        print "acceptances =", acceptances
        print "sigma_loga =", sigma_loga
        print "sigma_acceptance =", sigma_acceptance
pool.close()
pool.join()
MCMC = MCMC[burn:, :]

np.savetxt(mcmc_file, MCMC)
        
tt = time.time()-start
print "Time taken: {} s = {} min".format(round(tt), round(tt/60.,1))
#print final_state

with open(log_file, "w") as outfile:
    outfile.write("Model {}: {}\n".format(pyap_options["model_number"], model_name))
    outfile.write("Expt name: {}\n".format(expt_name))
    outfile.write("First trace: {}\n".format(trace_name))
    outfile.write("Fitting to {} traces\n\n".format(N_e))
    outfile.write("Total time taken: {} s = {} min = {} hr\n\n".format(round(tt), round(tt/60.,1), round(tt/3600.,2)))
    outfile.write("Final logas: {}\n".format(logas))
    outfile.write("Final sigma_loga: {}\n".format(sigma_loga))
    outfile.write("Final acceptances: {}\n".format(acceptances))
    outfile.write("Final sigma_acceptance: {}\n".format(sigma_acceptance))

print "\nAll done.\n"

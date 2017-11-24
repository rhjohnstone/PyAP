import ap_simulator  
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp
import argparse
import matplotlib.pyplot as plt

npsum = np.sum
npexp = np.exp
nplog = np.log

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data", required=True)
parser.add_argument("-i", "--iterations", type=int, help="total MCMC iterations", default=500000)
parser.add_argument("-s", "--seed", type=int, help="Python random seed", default=1)
parser.add_argument("--num-cores", type=int, help="number of cores for multiprocessing", default=1)
parser.add_argument("--cheat", action="store_true", help="for synthetic data: start MCMC from parameter values used to generate data", default=False)
parser.add_argument("--different", action="store_true", help="use different initial guess for some params", default=False)
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


sigma_uniform_lower = 1e-3
sigma_uniform_upper = 25.
omega = 0.5*nplog(10)  # s.d. of Normal priors on lnGs
two_omega_sq = 2.*omega**2


def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))


def solve_for_voltage_trace_without_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        sys.exit()


def solve_for_voltage_trace_with_null(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params[:-1]))
    except:
        print "\n\nFAIL\n\n"
        sys.exit()


def solve_for_voltage_trace_without_null(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        sys.exit()


if (pyap_options["model_number"]==666):
    cpp_model_number = 2
    solve_for_voltage_trace = solve_for_voltage_trace_with_null
else:
    cpp_model_number = pyap_options["model_number"]
    solve_for_voltage_trace = solve_for_voltage_trace_without_null
    

'''if data_clamp_on < data_clamp_off:
    solve_for_voltage_trace = solve_for_voltage_trace_with_initial_V
    print "Solving after setting V(0) = data(0)"
else:
    solve_for_voltage_trace = solve_for_voltage_trace_without_initial_V
    print "Solving without setting V(0) = data(0)"'''


def log_target(temp_params, ap_model, expt_trace):
    """Log target distribution with Normal prior for lnGs, uniform for sigma"""
    temp_lnGs, temp_sigma = temp_params[:-1], temp_params[-1]
    if not (sigma_uniform_lower < temp_sigma < sigma_uniform_upper):
        return -np.inf
    else:
        try:
            test_trace = solve_for_voltage_trace(temp_lnGs, ap_model, expt_trace)
        except:
            #print "Failed to solve at iteration", t
            print "exp(temp_lnGs):\n", exp(temp_lnGs)
            print "original_gs:\n", original_gs
            return -np.inf
        return - num_pts*nplog(temp_sigma) - npsum((test_trace-expt_trace)**2)/(2.*temp_sigma**2) - npsum((temp_lnGs-log_gs)**2)/two_omega_sq

    
def compute_initial_sigma(temp_lnGs, ap_model, expt_trace):
    test_trace = solve_for_voltage_trace(temp_lnGs, ap_model, expt_trace)
    #plt.plot(expt_trace)
    #plt.plot(test_trace)
    #plt.show()
    #sys.exit()
    return np.sqrt(npsum((test_trace-expt_trace)**2)/len(expt_trace))
    



def do_mcmc_adaptive(ap_model, expt_trace):
    initial_it_file = log_file[:-4]+"_initial_iteration.txt"
    npr.seed(args.seed)
    print "Starting chain"
    start = time.time()
    if not args.cheat:
        cmaes_unscaled = False  # only doing MCMC to CMA-ES fits done with exp scaling
        cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(pyap_options["model_number"], expt_name, trace_name, cmaes_unscaled)
        try:
            cmaes_results = np.loadtxt(cmaes_best_fits_file)
            ndim = cmaes_results.ndim
            if ndim == 1:
                best_gs = cmaes_results[:-1]
            else:
                best_index = np.argmin(cmaes_results[:,-1])
                best_gs = cmaes_results[best_index,:-1]
            initial_gs = best_gs
        except Exception, e:
            print "\n",e,"\n"
            initial_gs = original_gs
    else:
        trace_number = int(trace_path.split(".")[-2].split("_")[-1])
        cheat_params_file = '/'.join( split_trace_path[:5] ) + "/expt_params.txt"
        expt_gs = np.loadtxt(cheat_params_file)[trace_number, :]
        initial_gs = expt_gs
    initial_ln_gs = nplog(initial_gs)
    initial_sigma = compute_initial_sigma(initial_ln_gs, ap_model, expt_trace)
    if initial_sigma < sigma_uniform_lower:
        temp_sigma = sigma_uniform_lower + 1e-3
    elif initial_sigma > sigma_uniform_upper:
        temp_sigma = sigma_uniform_upper - 1e-3
    theta_cur = np.concatenate((initial_ln_gs,[initial_sigma]))
    if args.different:
        for jj in [9,10,11]:
            theta_cur[jj] = nplog(10. * original_gs[jj] * npr.rand())
    cov_estimate = 0.001*np.diag(theta_cur**2)
    print "\ntheta_cur:", theta_cur, "\n"
    print "\noriginal_gs:", original_gs, "\n"
    print "\nexp(theta_cur):", npexp(theta_cur), "\n"
    log_target_cur = log_target(theta_cur, ap_model, expt_trace)

    total_iterations = args.iterations
    thinning = 5
    num_saved = total_iterations / thinning + 1
    burn = num_saved / 4

    chain = np.zeros((num_saved, num_params+1))
    chain[0, :] = np.concatenate((theta_cur, [log_target_cur]))
    np.savetxt(initial_it_file, chain[0, :])

    loga = 0.
    acceptance = 0.    

    status_when = total_iterations / 100
    adapt_when = 100*num_params

    t = 1
    s = 1
    while t <= total_iterations:
        theta_star = npr.multivariate_normal(theta_cur, npexp(loga)*cov_estimate)
        """try:
            theta_star = npr.multivariate_normal(theta_cur, npexp(loga)*cov_estimate)
        except Warning as e:
            print str(e)
            print "Iteration:", t
            print "theta_cur:", theta_cur
            print "loga:", loga
            print "cov_estimate:", cov_estimate
            sys.exit()"""
        log_target_star = log_target(theta_star, ap_model, expt_trace)
        u = npr.rand()
        if nplog(u) < log_target_star - log_target_cur:
            accepted = 1
            theta_cur = theta_star
            log_target_cur = log_target_star
        else:
            accepted = 0
        acceptance = (t-1.)/t * acceptance + 1./t * accepted
        if t % thinning == 0:
            chain[t/thinning,:] = np.concatenate((theta_cur, [log_target_cur]))
        if t % status_when == 0:
            #pass
            print "{} / {}".format(t/status_when, total_iterations/status_when)
            print "acceptance =", acceptance
            time_taken_so_far = time.time()-start
            estimated_time_left = int(total_iterations*time_taken_so_far/t - time_taken_so_far)
            print "\n\nEstimated time remaining: {} s = {}-ish min\n\n".format(estimated_time_left, estimated_time_left/60)
        if t == adapt_when:
            mean_estimate = np.copy(theta_cur)
        if t > adapt_when:
            gamma_s = 1./(s+1.)**0.6
            temp_covariance_bit = np.array([theta_cur-mean_estimate])
            cov_estimate = (1-gamma_s) * cov_estimate + gamma_s * np.dot(np.transpose(temp_covariance_bit),temp_covariance_bit)
            mean_estimate = (1-gamma_s) * mean_estimate + gamma_s * theta_cur
            loga += gamma_s*(accepted-0.25)
            s += 1
        t += 1
    # discard burn-in before saving chain, just to save space mostly
    time_taken = int(time.time() - start)
    print "\n\nTime taken: {} s = {} min\n\n".format(time_taken,time_taken/60)
    return chain[burn:,:], loga, acceptance


do_mcmc = do_mcmc_adaptive

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)
#solve_end = 100  # just for HH
original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
log_gs = nplog(original_gs)
num_params = len(original_gs)+1  # include sigma


try:
    expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
except:
    sys.exit( "\n\nCan't find (or load) {}\n\n".format(trace_path) )

if args.different:
    expt_times = expt_times[::2]
    expt_trace = expt_trace[::2]
num_pts = len(expt_trace)
pi_bit = 0.5 * num_pts * nplog(2 * np.pi)

solve_start, solve_end = expt_times[[0,-1]]
solve_timestep = expt_times[1] - expt_times[0]

ap_model = ap_simulator.APSimulator()
if (data_clamp_on < data_clamp_off):
    ap_model.DefineStimulus(0, 1, 1000, 0)  # no injected stimulus current
    ap_model.DefineModel(cpp_model_number)
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
else:
    ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
    ap_model.DefineModel(cpp_model_number)
ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])


mcmc_file, log_file, png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, trace_name)
log_start_time = time.time()
chain, final_loga, final_acceptance = do_mcmc(ap_model, expt_trace)
num_saved = chain.shape[0]
log_time_taken = time.time() - log_start_time
np.savetxt(mcmc_file, chain)
with open(log_file, "w") as outfile:
    outfile.write("Expt: {}\n".format(expt_name))
    outfile.write("Trace: {}\n".format(trace_name))
    outfile.write("Saved iterations: {}\n".format(num_saved))
    outfile.write("Time taken: {} s = {} min = {} hr\n".format(int(log_time_taken), round(log_time_taken/60.,1), round(log_time_taken/3600.,1)))
    outfile.write("Final loga: {}\n".format(final_loga))
    outfile.write("Final acceptance rate: {}\n".format(final_acceptance))
print "\nSaved MCMC output at {}\n".format(mcmc_file)





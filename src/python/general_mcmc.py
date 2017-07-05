import ap_simulator  
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp


def exponential_scaling(unscaled_params):
    return original_gs ** unscaled_params


def solve_for_voltage_trace(temp_g_params, ap_model):
    ap_model.SetToModelInitialConditions()
    return ap_model.SolveForVoltageTraceWithParams(temp_g_params)


def log_target(temp_unscaled_params, ap_model, expt_trace):
    temp_unscaled_gs, temp_sigma = temp_unscaled_params[:-1], temp_unscaled_params[-1]
    if (temp_sigma <= 0):
        return -np.inf
    else:
        temp_gs = exponential_scaling(temp_unscaled_gs)
        try:
            test_trace = solve_for_voltage_trace(temp_gs, ap_model)
        except:
            print "Failed to solve"
            print temp_gs
            sys.exit()
        return -len(expt_trace)*np.log(temp_sigma) - np.sum((test_trace-expt_trace)**2)/(2.*temp_sigma**2) + np.dot(temp_unscaled_gs, log_gs)
    
    
def compute_initial_sigma(temp_unscaled_gs, ap_model, expt_trace):
    temp_gs = exponential_scaling(temp_unscaled_gs)
    test_trace = solve_for_voltage_trace(temp_gs, ap_model)
    return np.sqrt(np.sum((test_trace-expt_trace)**2)/len(expt_trace))


def do_mcmc(ap_model, expt_trace, temperature):#, theta0):
    #npr.seed(trace_number)
    print "Starting chain"
    start = time.time()
    cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(ps.pyap_options["model_number"])
    try:
        cmaes_results = np.loadtxt(cmaes_best_fits_file)
        ndim = cmaes_results.ndim
        if ndim == 1:
            best_gs = cmaes_results[:-1]
        else:
            best_index = np.argmin(cmaes_results[:,-1])
            best_gs = cmaes_results[best_index,:-1]
        initial_unscaled_gs = np.log(best_gs) / log_gs
    except Exception, e:
        print "\n",e,"\n"
        initial_unscaled_gs = np.ones(num_params-1)
    theta_cur = np.concatenate((initial_unscaled_gs,[compute_initial_sigma(initial_unscaled_gs, ap_model, expt_trace)]))
    print "\ntheta_cur:", theta_cur, "\n"
    log_target_cur = log_target(theta_cur, ap_model, expt_trace)

    total_iterations = 500000
    thinning = 5
    num_saved = total_iterations / thinning + 1
    burn = num_saved / 4

    chain = np.zeros((num_saved, num_params+1))
    chain[0, :] = np.concatenate((theta_cur, [log_target_cur]))

    loga = 0.
    acceptance = 0.

    mean_estimate = np.abs(theta_cur)
    cov_estimate = 0.0001*np.eye(num_params)

    status_when = total_iterations / 100
    adapt_when = 500*num_params

    t = 1
    s = 1
    while t <= total_iterations:
        theta_star = npr.multivariate_normal(theta_cur, np.exp(loga)*cov_estimate)
        """try:
            theta_star = npr.multivariate_normal(theta_cur, np.exp(loga)*cov_estimate)
        except Warning as e:
            print str(e)
            print "Iteration:", t
            print "temperature:", temperature
            print "theta_cur:", theta_cur
            print "loga:", loga
            print "cov_estimate:", cov_estimate
            sys.exit()"""
        log_target_star = log_target(theta_star, ap_model, expt_trace)
        u = npr.rand()
        if np.log(u) < log_target_star - log_target_cur:
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
            print t/status_when, "/", total_iterations/status_when
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
    chain = chain[burn:, :]
    chain[:,:-2] = original_gs**chain[:,:-2]  # return params scaled back into G-space
    return chain


stimulus_magnitude = 0
stimulus_duration = 1
stimulus_start_time = 0.
original_gs, g_parameters = ps.get_original_params(ps.pyap_options["model_number"])
log_gs = np.log(original_gs)
num_params = len(original_gs)+1  # include sigma


def do_everything():
    try:
        expt_times, expt_trace = np.loadtxt(ps.trace_path,delimiter=',').T
    except:
        sys.exit( "\n\nCan't find (or load) {}\n\n".format(ps.trace_path) )
    
    solve_start, solve_end = expt_times[[0,-1]]
    solve_timestep = expt_times[1] - expt_times[0]

    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, ps.pyap_options["stimulus_period"], stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start, solve_end, solve_timestep)
    ap_model.DefineModel(ps.pyap_options["model_number"])
    ap_model.SetExtracellularPotassiumConc(ps.pyap_options["extra_K_conc"])
    ap_model.SetIntracellularPotassiumConc(ps.pyap_options["intra_K_conc"])
    ap_model.SetExtracellularSodiumConc(ps.pyap_options["extra_Na_conc"])
    ap_model.SetIntracellularSodiumConc(ps.pyap_options["intra_Na_conc"])
    ap_model.SetNumberOfSolves(ps.pyap_options["num_solves"])
    ap_model.UseDataClamp(ps.pyap_options["data_clamp_on"], ps.pyap_options["data_clamp_off"])
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)

    temperature = 1
    mcmc_file, log_file, png_dir = ps.mcmc_exp_scaled_file_log_file_and_figs_dirs(ps.pyap_options["model_number"])
    log_start_time = time.time()
    chain = do_mcmc(ap_model, expt_trace, temperature)
    log_time_taken = time.time() - log_start_time
    np.savetxt(mcmc_file, chain)
    with open(log_file, "w") as outfile:
        outfile.write(ps.expt_name+"\n")
        outfile.write(ps.trace_name+"\n")
        outfile.write("Time taken: {} s = {} min = {} hr\n".format(int(log_time_taken), int(log_time_taken/60.), round(log_time_taken/3600.,1)))
    print "\nSaved MCMC output at {}\n".format(mcmc_file)
    return None
    


"""num_cores = 1  # 16 for arcus-b
pool = mp.Pool(num_cores)
mcms = pool.map_async(do_everything, traces).get(9999999)
pool.close()
pool.join()"""

do_everything()


"""fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(chain[:,0],color='blue',edgecolor='blue',normed=True,bins=40)
ax.legend()
plt.show(block=True)"""


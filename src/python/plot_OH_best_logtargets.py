import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import itertools as it
import numpy.random as npr
import argparse

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
parser.add_argument("--rel-tol", type=int, help="rel tol exponent", default=7)
parser.add_argument("-t", "--trace", type=int, help="expt trace", default=0)
args, unknown = parser.parse_known_args()


true_noise_sd = 0.5
two_sigma_sq = 2.*true_noise_sd**2
omega = 0.5*np.log(10)  # s.d. of Normal priors on lnGs
two_omega_sq = 2.*omega**2
    
def log_target(temp_params, ap_model, expt_trace):
    temp_lnGs, temp_sigma = temp_params[:-1], temp_params[-1]
    if not (sigma_uniform_lower < temp_sigma < sigma_uniform_upper):
        return -np.inf
    else:
        ap_model.SetToModelInitialConditions()
        temp_trace = ap.SolveForVoltageTraceWithParams(np.exp(temp_lnGs))
        return -num_pts*nplog(temp_sigma) - np.sum((test_trace-expt_trace)**2)/(2.*temp_sigma**2) - np.sum((temp_lnGs-log_gs)**2)/two_omega_sq

long_mcmc_best = [4.47540408283, -9.26912635885, -5.82717934, -1.70409472553, -2.99804276085, -5.87576855368, -6.95420558863, 3.09218137611, -3.89167278528, -17.2057808491, -22.0638029403, -7.14859640155, -5.20561556468, 0.481639397543]



abs_tol = args.rel_tol+2


protocol = 1
model_number = 5

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)

expt_file = "projects/PyAP/python/input/synthetic_ohara_lnG/traces/synthetic_ohara_lnG_trace_0.csv".format(args.trace)
expt_times, expt_trace = np.loadtxt(expt_file, delimiter=',').T
solve_start = expt_times[0]
solve_end = expt_times[-1]
solve_timestep = expt_times[1] - expt_times[0]
num_data_pts = len(expt_trace)

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
original_gs = np.array(original_gs)
log_gs = np.log(original_gs)

params_file = "projects/PyAP/python/input/synthetic_ohara_lnG/expt_params.txt"
expt_params = np.loadtxt(params_file)[args.trace,:]

ap = ap_simulator.APSimulator()
ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
ap.DefineModel(model_number)
ap.SetTolerances(10**-args.rel_tol, 10**-abs_tol)

ap.SetToModelInitialConditions()
model_trace = ap.SolveForVoltageTraceWithParams(original_gs)

print "best log-target from long MCMC:", log_target(long_mcmc_best, ap, expt_trace)
ap.SetToModelInitialConditions()
best_long_mcmc_ap = ap.SolveForVoltageTraceWithParams(np.exp(long_mcmc_best[:-1]))

ap.SetToModelInitialConditions()
true_trace = ap.SolveForVoltageTraceWithParams(expt_params)
print "log-target from true params:", log_target(np.concatenate((np.log(expt_params),[true_noise_sd])), ap, expt_trace)

param_idx = args.param_idx
print "\n{}\n".format(g_parameters[param_idx])


num_samples = args.num_pts
    
cs = ['#1b9e77','#d95f02','#7570b3']

    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("log($G_{pCa}$)")
ax.set_ylabel('Approx. log-target')
ax.grid()
ax.plot(expt_times, expt_trace, lw=2, color=cs[0])
ax.plot(expt_times, true_trace, lw=2, color=cs[1])
ax.plot(expt_times, best_long_mcmc_ap, lw=2, color=cs[2])
plt.show()
sys.exit()

phi = 1.61803398875
fig_y = 4

    

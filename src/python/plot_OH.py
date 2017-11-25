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
#requiredNamed.add_argument("--xmin", type=float, help="min scale for GpCa", required=True)
#requiredNamed.add_argument("--xmax", type=float, help="max scale for GpCa", required=True)
requiredNamed.add_argument("-n", "--num-pts", type=int, help="number of points to plot", required=True)
parser.add_argument("--rel-tol", type=int, help="rel tol exponent", default=7)
args, unknown = parser.parse_known_args()


true_noise_sd = 0.5
two_sigma_sq = 2.*true_noise_sd**2
    
def approx_likelihood(test_trace):
    return -np.log(true_noise_sd) - np.sum((expt_trace-test_trace)**2)/two_sigma_sq


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular) (not available atm)
# 8. Gokhale 2017 ex293 (not available atm)

abs_tol = args.rel_tol+2


protocol = 1
model_number = 5

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)

expt_file = "projects/PyAP/python/input/synthetic_ohara_lnG/traces/synthetic_ohara_lnG_trace_0.csv"
expt_times, expt_trace = np.loadtxt(expt_file, delimiter=',').T
solve_start = expt_times[0]
solve_end = expt_times[-1]
solve_timestep = expt_times[1] - expt_times[0]

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
original_gs = np.array(original_gs)

params_file = "projects/PyAP/python/input/synthetic_ohara_lnG/expt_params.txt"
expt_params = np.loadtxt(params_file)[0,:]

ap = ap_simulator.APSimulator()
ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
ap.DefineModel(model_number)
ap.SetTolerances(10**-args.rel_tol, 10**-abs_tol)

ap.SetToModelInitialConditions()
model_trace = ap.SolveForVoltageTraceWithParams(original_gs)

ap.SetToModelInitialConditions()
true_trace = ap.SolveForVoltageTraceWithParams(expt_params)

true_aprox_ll = approx_likelihood(true_trace)

print "approx_likelihood(model_trace) =", approx_likelihood(model_trace)
print "approx_likelihood(true_trace) =", true_aprox_ll

G_pCa = expt_params[11]

num_samples = args.num_pts
#xmin = args.xmin
#xmax = args.xmax

"""y = np.zeros(num_samples)
x = np.linspace(xmin, xmax, num_samples)
temp_params = np.copy(expt_params)
for i, scale in enumerate(x):
    temp_params[11] = scale*G_pCa
    ap.SetToModelInitialConditions()
    temp_trace = ap.SolveForVoltageTraceWithParams(temp_params)
    y[i] = approx_likelihood(temp_trace)

z = np.zeros(num_samples)
wmin = np.log10(xmin)
wmax = np.log10(xmax)
w = np.logspace(wmin, wmax, num_samples)
temp_params = np.copy(expt_params)
for i, scale in enumerate(w):
    temp_params[11] = scale * G_pCa
    ap.SetToModelInitialConditions()
    temp_trace = ap.SolveForVoltageTraceWithParams(temp_params)
    z[i] = approx_likelihood(temp_trace)"""
    
cs = ['#1b9e77','#d95f02','#7570b3']

x = np.linspace(-9, -5.5, num_samples)
y = np.zeros(num_samples)
temp_params = np.copy(expt_params)
for i, log_param in enumerate(x):
    temp_params[11] = np.exp(log_param)
    ap.SetToModelInitialConditions()
    temp_trace = ap.SolveForVoltageTraceWithParams(temp_params)
    y[i] = approx_likelihood(temp_trace)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("log($G_{pCa}$)")
ax.set_ylabel('Approx. log-likelihood')
ax.grid()
ax.plot(x, y, lw=2, color=cs[0])
ax.set_xlim(xmin, xmax)
ax.axvline(np.log(G_pCa), lw=2, color=cs[1])
plt.show()
sys.exit()

phi = 1.61803398875
fig_y = 4


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(phi*fig_y, fig_y))
ax1.set_xlabel("$G_{pCa} / G_{pCa,true}$")
ax1.set_ylabel('Approx. log-likelihood')
ax1.grid()
ax1.plot(x, y, lw=2, color=cs[0])
ax1.set_xlim(xmin, xmax)
ax1.axvline(1, lw=2, color=cs[1])
ax1.axhline(true_aprox_ll, color=cs[2])
ax1.set_xticks([xmin] + list(ax1.get_xticks())[1:])

ax2.set_xlabel(r"$\log_{10} (G_{pCa} / G_{pCa,true})$")
ax2.grid()
ax2.plot(w, z, lw=2, color=cs[0])
ax2.set_xscale('log')
ax2.set_xlim(xmin, xmax)
ax2.axvline(1, lw=2, color=cs[1])
ax2.axhline(true_aprox_ll, color=cs[2])
ax2.set_xlim(xmin, xmax)

fig.tight_layout()
plt.show()


"""fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane voltage (mV)')
ax.grid()
ax.plot(expt_times, expt_trace, lw=2, label='Expt')
ax.plot(expt_times, model_trace, lw=2, label='Model')
ax.plot(expt_times, true_trace, lw=2, label='True')
ax.set_title(model_name)
ax.legend()
plt.show(block=True)"""
    
    

import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import itertools as it
import numpy.random as npr



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

ap.SetToModelInitialConditions()
model_trace = ap.SolveForVoltageTraceWithParams(original_gs)

ap.SetToModelInitialConditions()
true_trace = ap.SolveForVoltageTraceWithParams(expt_params)

print "approx_likelihood(model_trace) =", approx_likelihood(model_trace)
print "approx_likelihood(true_trace) =", approx_likelihood(true_trace)

G_pCa = expt_params[11]
num_samples = 101

y = np.zeros(num_samples)
x = np.linspace(0, 10, num_samples)
temp_params = np.copy(expt_params)
for i, scale in enumerate(x):
    temp_params[11] = scale*G_pCa
    ap.SetToModelInitialConditions()
    temp_trace = ap.SolveForVoltageTraceWithParams(temp_params)
    y[i] = approx_likelihood(temp_trace)

z = np.zeros(num_samples)
wmin = np.log(0.1)
wmax = np.log(10)
w = np.logspace(wmin, wmax, num_samples, base=np.exp(1))
print w
temp_params = np.copy(expt_params)
for i, scale in enumerate(w):
    temp_params[11] = scale * G_pCa
    ap.SetToModelInitialConditions()
    temp_trace = ap.SolveForVoltageTraceWithParams(temp_params)
    z[i] = approx_likelihood(temp_trace)
    
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_xlabel("$G_{pCa} / G_{pCa,true}$")
ax1.set_ylabel('Approx. log-likelihood')
ax1.grid()
ax1.plot(x,y)
ax1.axvline(1, lw=2, color='green')

ax2.set_xlabel(r"$\log (G_{pCa} / G_{pCa,true})$")
ax2.grid()
ax2.plot(w, z)
ax2.set_xscale('log', base=np.exp(1))
ax2.set_scale(0, 10)
ax2.axvline(0, lw=2, color='green')

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
    
    

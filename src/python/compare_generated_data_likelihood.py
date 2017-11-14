import ap_simulator
import numpy as np
import numpy.random as npr
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import os
import argparse


def solve_for_voltage_trace(temp_g_params):
    ap_model.SetToModelInitialConditions()
    return ap_model.SolveForVoltageTraceWithParams(temp_g_params)


def log_likelihood(params):
    temp_g_params, temp_sigma = params[:-1], params[-1]
    print "temp_g_params:", temp_g_params
    print "temp_sigma =", temp_sigma
    temp_trace = solve_for_voltage_trace(temp_g_params)
    return -len(temp_trace)*np.log(temp_sigma) - np.sum((temp_trace-expt_trace)**2)/(2*temp_sigma**2)


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("-m", "--model", type=int, help="AP model number", required=True)
args, unknown = parser.parse_known_args()

# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.grid()
#ax.set_xlabel("Time (ms)")
#ax.set_ylabel("Membrane voltage (mV)")

python_seed = 1
npr.seed(python_seed)

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)

model = args.model

if model==1:
    label="hodgkin_huxley"
    expt_params_normal_sd = 0.15
elif model==2:
    label = "beeler_reuter"
    expt_params_normal_sd = 0.15
elif model==3:
    label = "luo_rudy"
    expt_params_normal_sd = 0.1
elif model==4:
    label = "ten_tusscher"
    expt_params_normal_sd = 0.3
elif model==5:
    label = "ohara"
    expt_params_normal_sd = 0.2
elif model==6:
    label = "davies"
    expt_params_normal_sd = 0.11
elif model==7:
    label = "paci"
    expt_params_normal_sd = 0.1

print label

expt_name = "synthetic_{}".format(label)
expt_dir = "../workspace/PyAP/src/python/input/{}/".format(expt_name)
traces_dir = "{}traces/".format(expt_dir)
if not os.path.exists(traces_dir):
    os.makedirs(traces_dir)
options_file = "{}PyAP_options.txt".format(expt_dir)
expt_params_file = "{}expt_params.txt".format(expt_dir)

noise_sigma = 0.25


pyap_options = { "model_number":model,
                 "num_solves":1,
                 "extra_K_conc":5.4,
                 "intra_K_conc":130,
                 "extra_Na_conc":140,
                 "intra_Na_conc":10,
                 "data_clamp_on":0,
                 "data_clamp_off":0,
                 "stimulus_period":1000 }

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)

#expt_params = original_gs * (1. + 0.1*npr.randn(len(original_gs)))
#expt_params[np.where(expt_params<0.)] = 0.

expt_params_mean = original_gs

num_expts = 32

all_expt_params = (1. + expt_params_normal_sd*npr.randn(num_expts, num_gs)) * original_gs
print all_expt_params

if np.any(all_expt_params<0):
    sys.exit("Some negative parameter values generated.\n")



expt_times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
ap_model.DefineModel(pyap_options["model_number"])
ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])
ap_model.SetTolerances(1e-10, 1e-12)

i = 0
g_params = all_expt_params[i, :]
print "g_params:", g_params
real_bit = solve_for_voltage_trace(g_params)
expt_trace = real_bit + noise_sigma*npr.randn(len(expt_times))
true_params = np.concatenate((g_params, [noise_sigma]))
print "true_params:", true_params
print log_likelihood(true_params)

best_params = np.array([  9.93691193e+01,   8.73592563e-05,   2.54699743e-03,   1.51319231e-01,
   5.32600004e-02,   1.61116185e-03,   1.07445324e-03,   2.46801212e+01,
   2.11477010e-02,   5.73068132e-09,   8.57302566e-10,   1.74340157e-02,
   5.96320573e-03,   2.51272811e-01])
best_ll_trace = solve_for_voltage_trace(best_params[:-1])
print log_likelihood(best_params)

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane voltage (mV)')
ax.set_title(model_name)
ax.grid()
ax.plot(expt_times, expt_trace, label='expt')
ax.plot(expt_times, real_bit, label='real')
ax.plot(expt_times, best_ll_trace, label='best ll')
ax.legend()
fig.tight_layout()
#fig.savefig(expt_dir+"{}_synthetic_expt_traces.png".format(model_name))
#fig.savefig(expt_dir+"{}_synthetic_expt_traces.pdf".format(model_name))
plt.show(block=True)



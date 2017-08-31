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


def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    try:
        return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    except ap_simulator.CPPException, e:
        print e.GetShortMessage
        print "temp_g_params:\n", temp_g_params
        print "original_gs:\n", original_gs
        return np.zeros(len(expt_times))


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

model = 4

if model==1:
    label="hodgkin_huxley"
elif model==2:
    label = "beeler_reuter"
elif model==3:
    label = "luo_rudy"
elif model==4:
    label = "ten_tusscher"
elif model==5:
    label = "ohara"
elif model==6:
    label = "davies"


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
                 
with open(options_file, "w") as outfile:
    for option in pyap_options:
        outfile.write('{} {}\n'.format(option, pyap_options[option]))

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(original_gs)

#expt_params = original_gs * (1. + 0.1*npr.randn(len(original_gs)))
#expt_params[np.where(expt_params<0.)] = 0.

expt_params_mean = original_gs
expt_params_normal_sd = 0.2
num_expts = 32

all_expt_params = (1. + expt_params_normal_sd*npr.randn(num_expts, num_gs)) * original_gs
print all_expt_params

with open(expt_params_file, "w") as outfile:
    outfile.write("# synthetic data experimental parameter values\n")
    outfile.write("# {}\n".format(model_name))
    outfile.write("# {} sets of parameter values\n".format(num_expts))
    outfile.write("# standard deviation for Normally-generated parameters: {}\n".format(expt_params_normal_sd))
    np.savetxt(outfile, all_expt_params)

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

fig = plt.figure()
ax = fig.add_subplot(111)

for i in xrange(num_expts):
    expt_trace = solve_for_voltage_trace(all_expt_params[i, :], ap_model) + noise_sigma*npr.randn(len(expt_times))
    np.savetxt(traces_dir+expt_name+"trace_{}.csv".format(i), np.vstack((expt_times, expt_trace)).T)
    ax.plot(expt_times, expt_trace)

fig.tight_layout()
plt.show()



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

# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293
# 9. Davies (canine) linearised by RJ
# 10. Paci linearised by RJ

python_seed = 1
for python_seed in xrange(1,4):
    npr.seed(python_seed)

    expt_name = "synthetic_davies"
    trace_name = "synthetic_davies_seed_{}".format(python_seed)
    traces_dir = "../workspace/PyAP/src/python/input/{}/traces/".format(expt_name)
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)
    trace_file = traces_dir+"{}.csv".format(trace_name)
    options_file = "../workspace/PyAP/src/python/input/{}/PyAP_options.txt".format(expt_name)


    noise_sigma = 1.

    solve_start = 0.
    solve_end = 500.
    solve_timestep = 0.2
    stimulus_magnitude = -25.5
    stimulus_duration = 2
    stimulus_start_time = 50

    pyap_options = { "model_number":9,
                     "num_solves":2,
                     "extra_K_conc":5.4,
                     "intra_K_conc":130,
                     "extra_Na_conc":140,
                     "intra_Na_conc":10,
                     "data_clamp_on":50,
                     "data_clamp_off":52,
                     "stimulus_period":1000 }
                     
    with open(options_file, "w") as outfile:
        for option in pyap_options:
            outfile.write('{} {}\n'.format(option, pyap_options[option]))

    original_gs, g_parameters = ps.get_original_params(pyap_options["model_number"])

    expt_params = original_gs * (1. + 0.01*npr.randn(len(original_gs)))
    expt_params[np.where(expt_params<0.)] = 0.

    times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

    ap_model = ap_simulator.APSimulator()
    ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
    ap_model.DefineSolveTimes(solve_start, solve_end, solve_timestep)
    ap_model.DefineModel(pyap_options["model_number"])
    ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
    ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
    ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
    ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
    ap_model.SetNumberOfSolves(pyap_options["num_solves"])
    try:
        print "Going to try to solve with params"
        true_trace = ap_model.SolveForVoltageTraceWithParams(original_gs)
    except ap_simulator.CPPException as e:
        print e.GetMessage
        sys.exit()
        
    true_trace += noise_sigma*npr.randn(len(true_trace))
    np.savetxt(trace_file, np.vstack((times, true_trace)).T, delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times,true_trace)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane voltage (mV)")
    ax.set_title("Model {}".format(pyap_options["model_number"]))
plt.show(block=True)


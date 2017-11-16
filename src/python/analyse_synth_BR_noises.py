import pyap_setup as ps
import matplotlib.pyplot as plt
import numpy as np



model_number = 2

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
num_gs = len(original_gs)

expt_name = "synth_BR_different_noises"
unscaled = False

num_traces = 32

normalised_differences = np.zeros((num_traces, num_gs))

for t in xrange(num_traces):
    print "Trace", t
    trace_name = "synth_BR_different_noises_trace_{}".format(t)
    cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(model_number, expt_name, trace_name, unscaled)
    all_fits = np.loadtxt(cmaes_best_fits_file)
    best_fit_idx = np.argmin(all_fits[:, -1])
    best_params = all_fits[best_fit_idx, :-1]
    print "best_params =", best_params

    diff_vector = best_params - original_gs
    print "diff_vector =", diff_vector

    normalied_diff_vector = diff_vector/original_gs
    print "normalied_diff_vector =", normalied_diff_vector, "\n"
    normalised_differences[t, :] = normalied_diff_vector
    
print normalised_differences


import pyap_setup as ps
import matplotlib.pyplot as plt
import numpy as np

t = 0

model_number = 2

original_gs, g_parameters, model_name = ps.get_original_params(model_number)
num_gs = len(original_gs)

expt_name = "synth_BR_different_noises"
trace_name = "synth_BR_different_noises_trace_{}".format(t)
unscaled = False
cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(model_number, expt_name, trace_name, unscaled)
all_fits = np.loadtxt(cmaes_best_fits_file)
best_fit_idx = np.argmin(all_fits[:, -1])
best_params = all_fits[best_fit_idx, :-1]
print best_params


import pyap_setup as ps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

array = np.array

model_number = 5
original_gs, g_parameters, model_name = ps.get_original_params(model_number)
parameters = g_parameters + ["sigma"]
true_sigma = 0.25
true_params = np.concatenate((original_gs, [true_sigma]))
num_params = len(true_params)
#expt_name = "synth_BR_different_noises"
expt_name = "synthetic_ohara"
unscaled = True
num_traces = 32
non_adaptive = False
temperature = 1
normalised_differences = []

for t in xrange(num_traces):
    print "Trace", t
    trace_name = "{}_trace_{}".format(expt_name, t)
    mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(model_number, expt_name, trace_name, unscaled, non_adaptive, temperature)
    try:
        mcmc = np.loadtxt(mcmc_file)
    except:
        print "Can't load trace", t
        continue
    best_ll_idx = np.argmax(mcmc[:, -1])
    best_params = mcmc[best_ll_idx, :-1]
    diff_vector = best_params - true_params
    normalied_diff_vector = diff_vector/true_params
    normalised_differences.append(normalied_diff_vector)

num_saved = len(normalised_differences)
normalised_differences = np.array(normalised_differences)
    
data = {parameters[i] : normalised_differences[:, i] for i in xrange(num_params)}

"""data = {'G_{CaL}': array([ 0.08151781, -0.07135147,  0.02393881, -0.01293104, -0.02390832,
       -0.01028615, -0.04942092, -0.02065425, -0.04786993, -0.01294899,
        0.04221833,  0.02861734,  0.00725457,  0.0059484 ,  0.02334442,
        0.00358735,  0.03075899,  0.01810479, -0.01022092, -0.03078841,
       -0.01872701, -0.03560693,  0.14218796,  0.0437548 ,  0.01989413,
       -0.03373741,  0.04337694,  0.00030563,  0.01984794, -0.03328998,
        0.0817418 ,  0.00236785]), 'G_{K1}': array([ 0.05978187, -0.05856541,  0.03021602, -0.01819915, -0.02040972,
        0.01564312, -0.0376922 , -0.01218419, -0.02318461, -0.00239845,
        0.03908071,  0.02895723,  0.00339936, -0.00498583,  0.00909369,
        0.01570235,  0.02199024,  0.01130715,  0.00126586, -0.03516785,
       -0.00584002, -0.02472012,  0.1068966 ,  0.01247409,  0.02747975,
       -0.02974739,  0.03323984,  0.01095694,  0.00181775, -0.02566947,
        0.05904601, -0.00661653]), 'G_{Na}': array([ 0.04860776, -0.02036554,  0.03353898, -0.01924738,  0.03299797,
        0.05846468,  0.03221575, -0.04721491, -0.03138731, -0.00186692,
        0.10639568,  0.02559719,  0.00373574, -0.01624562,  0.00768275,
        0.02109949,  0.02576722,  0.00650371,  0.05210302, -0.04387104,
        0.0311498 , -0.07701824,  0.08528074, -0.01668513,  0.01648534,
       -0.00472389, -0.01138731,  0.03931443, -0.03597714, -0.00850633,
        0.0218331 , -0.03721507]), 'G_K': array([ 0.05501316, -0.04244488, -0.01073663,  0.0060297 , -0.01525778,
       -0.06182677, -0.03624433, -0.02276905, -0.0622288 , -0.03228659,
        0.01054194,  0.00828502,  0.00793704,  0.02340901,  0.03636757,
       -0.02992257,  0.02656623,  0.01097047, -0.0264827 ,  0.01186881,
       -0.03848898, -0.03213729,  0.09622162,  0.07859425, -0.01818396,
       -0.00699817,  0.03160743, -0.02488916,  0.04291807, -0.01921263,
        0.05907963,  0.02301807])}"""
        
df = pd.DataFrame.from_dict(data, orient='index')
df.index.rename('Parameter', inplace=True)

stacked = df.stack().reset_index()
stacked.rename(columns={'level_1': 'Trace', 0: 'Normalised difference vector component'}, inplace=True)

sns.stripplot(data=stacked, x='Parameter', y='Normalised difference vector component', jitter=True)
plt.show()


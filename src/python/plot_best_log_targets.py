import pyap_setup as ps
import ap_simulator
import argparse
import numpy as np
import sys
import numpy.random as npr
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import norm
from scipy.stats import invgamma

npexp = np.exp

"""def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    temp_Gs = npexp(temp_lnG_params)
    solved = ap_model.SolveForVoltageTraceWithParams(temp_Gs)
    #if solved[-1] > -70:
    #    print temp_Gs
    return solved"""


#python_seed = 1
#npr.seed(python_seed)

randn = npr.randn
sqrt = np.sqrt
def sample_from_N_IG(eta):
    mu, nu, alpha, beta = eta
    sigma_squared_sample = invgamma.rvs(alpha,scale=beta)
    sample = mu + sqrt(sigma_squared_sample/nu)*randn()
    return sample, sigma_squared_sample

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("-n", "--num-traces", type=int, help="which hMCMC to use", required=True)

args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"
expt_params_file = '/'.join( split_trace_path[:5] ) + "/expt_params.txt"

print expt_name

pyap_options = {}
with open(options_file, 'r') as infile:
    for line in infile:
        (key, val) = line.split()
        if (key == "model_number") or (key == "num_solves"):
            val = int(val)
        else:
            val = float(val)
        pyap_options[key] = val
        
data_clamp_on = pyap_options["data_clamp_on"]
data_clamp_off = pyap_options["data_clamp_off"]

# convert Cm and Istim to correct units, model-specific
if pyap_options["model_number"]==3:  # LR
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] * 1e-6
elif pyap_options["model_number"]==4:  # TT
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
    indices_to_keep = [0, 1, 2, 3, 4, 6]  # which currents we are fitting
elif pyap_options["model_number"]==5:  # OH
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-6
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-6
elif pyap_options["model_number"]==7:  # Pa
    Cm = pyap_options["membrane_capacitance_pF"] * 1e-12
    stimulus_magnitude = -pyap_options["stimulus_magnitude_pA"] / Cm * 1e-12
    indices_to_keep = [0, 1, 2, 3, 4, 9, 11]  # which currents we are fitting
num_params_to_fit = len(indices_to_keep) + 1  # +1 for sigma

original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
num_gs = len(indices_to_keep)
log_gs = np.log(original_gs[indices_to_keep])



temp_Gs = np.copy(original_gs)
def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    
    temp_Gs[indices_to_keep] = npexp(temp_lnG_params)
    try:
        return ap_model.SolveForVoltageTraceWithParams(temp_Gs)
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)


split_trace_name = trace_name.split("_")
if pyap_options["model_number"]==4 or pyap_options["model_number"]==7:
    first_trace_number = int(split_trace_name[-2])
print "first_trace_number:", first_trace_number
        
#original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
#num_gs = len(original_gs)

#g_labels = ["${}$".format(g) for g in g_parameters]

    
N_e = args.num_traces

color_idx = np.linspace(0, 1, N_e)


m_true = np.log(original_gs)
sigma2_true = 0.01

sigma_lower = 1e-3
sigma_upper = 25

sigma_const = 1./(sigma_upper-sigma_lower)

mu = m_true
alpha = 4.*np.ones(num_gs)
beta = (alpha+1.) * 0.04
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

prior_means = np.log(original_gs)
prior_sd = 0.5*np.log(10)

old_eta_js = np.vstack((mu, nu, alpha, beta)).T

expt_traces = []
MPDs = np.zeros((N_e, num_gs))
for n in xrange(N_e):
    temp_trace_number = first_trace_number + n
    if pyap_options["model_number"]==6:
        temp_trace_name = "_".join(split_trace_name[:-1])+"_"+str(temp_trace_number)
    elif pyap_options["model_number"]==4 or pyap_options["model_number"]==7:
        temp_trace_name = "_".join(split_trace_name[:-2])+"_{}_1".format(temp_trace_number)
    print temp_trace_name
    sl_mcmc_file, sl_log_file, sl_png_dir = ps.mcmc_lnG_file_log_file_and_figs_dirs(pyap_options["model_number"], expt_name, temp_trace_name)
    try:
        sl_chain = np.loadtxt(sl_mcmc_file)
        saved_its = sl_chain.shape[0]
        sl_chain = sl_chain[saved_its/4:, :]  # some chains take ages to converge
        #plt.plot(sl_chain[:,-1])
        #plt.show()
        #plt.close()
        max_target_idx = np.argmax(sl_chain[:,-1])
        MPDs[n, :] = sl_chain[max_target_idx, :-2]
    except:
        print "Can't load", sl_mcmc_file
        print "So lnG = 0 in later plotting"    
    
    if pyap_options["model_number"]==4:
        plot_trace_number = 100 + n
        plot_trace_path = "projects/PyAP/python/input/roche_ten_tusscher_correct_units/traces/Trace_2_2_{}_1.csv".format(plot_trace_number)
    elif pyap_options["model_number"]==7:
        plot_trace_number = 100 + n
        plot_trace_path = "projects/PyAP/python/input/roche_paci_correct_units/traces/Trace_2_2_{}_1.csv".format(plot_trace_number)
    #elif pyap_options["model_number"]==6:
    #    plot_trace_number = 150 + n
    #    plot_trace_path = "projects/PyAP/python/input/dog_teun_davies/traces/dog_AP_trace_{}.csv".format(plot_trace_number)
    expt_times, expt_trace = np.loadtxt(plot_trace_path, delimiter=',').T
    expt_traces.append(expt_trace)

ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(stimulus_magnitude, pyap_options["stimulus_duration_ms"], pyap_options["stimulus_period_ms"], pyap_options["stimulus_start_ms"])
ap_model.DefineModel(pyap_options["model_number"])
if (data_clamp_on < data_clamp_off):
    ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
    ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)        
ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
ap_model.SetNumberOfSolves(pyap_options["num_solves"])
ap_model.SetMembraneCapacitance(Cm)

for i in xrange(N_e):
    fig, ax = plt.subplots(1, 1)
    ax.grid()
    ax.set_title("Trace {}".format(100+i))
    ax.plot(expt_times, expt_traces[i], color='blue')
    ax.plot(expt_times, solve_for_voltage_trace_with_initial_V(MPDs[i, :], ap_model, expt_traces[0]), color='red')
    plt.show()







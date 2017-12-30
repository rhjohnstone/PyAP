#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ap_simulator
import numpy as np
import time
import pyap_setup as ps
import sys


def solve_for_voltage_trace_with_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    ap_model.SetVoltage(expt_trace[0])
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)


def solve_for_voltage_trace_without_initial_V(temp_lnG_params, ap_model, expt_trace):
    ap_model.SetToModelInitialConditions()
    try:
        return ap_model.SolveForVoltageTraceWithParams(npexp(temp_lnG_params))
    except:
        print "\n\nFAIL\n\n"
        return np.zeros(num_pts)
        

def compute_bic(k, ll):
    return np.log(num_pts)*k - 2*ll


npexp = np.exp
nplog = np.log
npinf = np.inf
npsum = np.sum
npsqrt = np.sqrt

sigma_uniform_lower = 1e-3
sigma_uniform_upper = 25.
omega = 0.5*nplog(10)  # s.d. of Normal priors on lnGs
two_omega_sq = 2.*omega**2

trace_numbers = range(100, 116)
models = ["luo_rudy", "ten_tusscher", "ohara", "paci"]

num_models = len(models)

best_APs = []
best_sigmas = []
expt_traces = []
best_lls = []
data_files = []

all_BICs = []

print "\n"
for t in trace_numbers:
    BICs = {}
    for m in models:
        trace_path = "projects/PyAP/python/input/roche_{}_correct_units/traces/Trace_2_2_{}_1.csv".format(m, t)
        split_trace_path = trace_path.split('/')
        expt_name = split_trace_path[4]
        trace_name = split_trace_path[-1][:-4]
        options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"
        

        pyap_options = {}
        with open(options_file, 'r') as infile:
            for line in infile:
                (key, val) = line.split()
                if (key == "model_number") or (key == "num_solves"):
                    val = int(val)
                else:
                    val = float(val)
                pyap_options[key] = val
                
        cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_log_likelihood_lnG(pyap_options["model_number"], expt_name, trace_name)

        print "\n", cmaes_best_fits_file
        try:
            cmaes_output = np.loadtxt(cmaes_best_fits_file)
            best_idx = np.argmax(cmaes_output[:, -1])
            best_params = cmaes_output[best_idx, :-1]
            best_ll = cmaes_output[best_idx, -1]
        except:  # only compute BICs for ones that have been log-likelihood CMA-ESed
            print "Can't find a CMA-ES output for {} Trace {}".format(m, t)
            continue
            
        best_sigmas.append(best_params[-1])
        best_lls.append(best_ll)

        expt_times, expt_trace = np.loadtxt(trace_path,delimiter=',').T
            
        num_pts = len(expt_trace)
        
        """expt_traces.append(np.copy(expt_trace))
                
        data_clamp_on = pyap_options["data_clamp_on"]
        data_clamp_off = pyap_options["data_clamp_off"]

        ap_model = ap_simulator.APSimulator()
        if (data_clamp_on < data_clamp_off):
            ap_model.DefineStimulus(0, 1, 1000, 0)  # no injected stimulus current
            ap_model.DefineModel(pyap_options["model_number"])
            ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
            ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)
        else:
            ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, pyap_options["stimulus_period"], stimulus_start_time)
            ap_model.DefineModel(pyap_options["model_number"])
        ap_model.DefineSolveTimes(expt_times[0], expt_times[-1], expt_times[1]-expt_times[0])
        ap_model.SetExtracellularPotassiumConc(pyap_options["extra_K_conc"])
        ap_model.SetIntracellularPotassiumConc(pyap_options["intra_K_conc"])
        ap_model.SetExtracellularSodiumConc(pyap_options["extra_Na_conc"])
        ap_model.SetIntracellularSodiumConc(pyap_options["intra_Na_conc"])
        ap_model.SetNumberOfSolves(pyap_options["num_solves"])
        if (data_clamp_on < data_clamp_off):
            ap_model.UseDataClamp(data_clamp_on, data_clamp_off)
            ap_model.SetExperimentalTraceAndTimesForDataClamp(expt_times, expt_trace)


        if data_clamp_on < data_clamp_off:
            solve_for_voltage_trace = solve_for_voltage_trace_with_initial_V
        else:
            solve_for_voltage_trace = solve_for_voltage_trace_without_initial_V"""


        all_time_start = time.time()
            
        original_gs, g_parameters, model_name = ps.get_original_params(pyap_options["model_number"])
        
        
        num_gs = len(original_gs)
        num_params = num_gs + 1
        #log_gs = nplog(original_gs)





        #best_AP = solve_for_voltage_trace(nplog(best_params[:-1]), ap_model, expt_trace)
        #best_APs.append(np.copy(best_AP))
        
        BIC = compute_bic(num_params, best_ll)
        BICs[m] = BIC
    
    if len(BICs)>0:
        all_BICs.append(BICs)

# now to print a \LaTeX table...
print "\n\n"
print r"\begin{tabular}{*{" + str(num_models+1) + "}{c|}}"
line = " & " + " & ".join(models) + r" \\"
print line
print r"\midrule"
#print models
for i, x in enumerate(all_BICs):
    temp_BICs = [int(x[m]) for m in models]
    min_idx = np.argmin(temp_BICs)
    max_idx = np.argmax(temp_BICs)
    stuff = [str(i+100)] + [str(int(x[m])) for m in models]  # just to ensure they're printed in the same order
    stuff[min_idx+1] = r"\cellcolor{green!25}" + stuff[min_idx+1]
    stuff[max_idx+1] = r"\cellcolor{red!25}" + stuff[max_idx+1]
    line = " & ".join(stuff) + r" \\"
    print line
print r"\bottomrule"
print r"\end{tabular}"
print "\n\n"


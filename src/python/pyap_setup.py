"""
CHOICE OF CELL MODEL 

# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293

"""

import numpy as np
import os
import sys
import socket

if socket.getfqdn().endswith("arcus.arc.local"):
    arcus_b = True
    print "\nHopefully on arcus-b\n"
else:
    arcus_b = False


def cmaes_and_figs_files(model_number, expt_name, trace_name):
    if arcus_b:
        cmaes_dir = os.path.expandvars("$DATA/PyAP_output/{}/cmaes/model_{}/".format(expt_name, model_number))
    else:
        cmaes_dir = "projects/PyAP/python/output/{}/cmaes/model_{}/".format(expt_name, model_number)
    txt_dir, png_dir, svg_dir = cmaes_dir+"params/", cmaes_dir+"figs/png/", cmaes_dir+"figs/svg/"
    for d in [txt_dir, png_dir, svg_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    cmaes_best_fits_file = txt_dir+"{}_model_{}_trace_{}_cmaes_best_fits.txt".format(expt_name, model_number, trace_name)
    best_fit_png = png_dir+"{}_model_{}_trace_{}_cmaes_best_fit.png".format(expt_name, model_number, trace_name)
    best_fit_svg = svg_dir+"{}_model_{}_trace_{}_cmaes_best_fit.svg".format(expt_name, model_number, trace_name)
    return cmaes_best_fits_file, best_fit_png, best_fit_svg


def mcmc_file_log_file_and_figs_dirs(model_number, expt_name, trace_name, unscaled, non_adaptive):
    if unscaled:
        scale_bit = "mcmc_unscaled"
    else:
        scale_bit = "mcmc_exp_scaled"
    if non_adaptive:
        adaptive_bit = "non_adaptive"
    else:
        adaptive_bit = "adaptive"
    if arcus_b:
        mcmc_dir = os.path.expandvars("$DATA/PyAP_output/{}/{}/{}/model_{}/{}/".format(expt_name, scale_bit, adaptive_bit, model_number, trace_name))
    else:
        mcmc_dir = "projects/PyAP/python/output/{}/{}/{}/model_{}/{}/".format(expt_name, scale_bit, adaptive_bit, model_number, trace_name)
    txt_dir, png_dir = mcmc_dir+"chain/", mcmc_dir+"figs/png/"
    for d in [txt_dir, png_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    mcmc_file = txt_dir+"{}_model_{}_trace_{}_{}_{}.txt".format(expt_name, model_number, trace_name, scale_bit, adaptive_bit)
    log_file = mcmc_dir+"{}_model_{}_trace_{}_{}_{}.log".format(expt_name, model_number, trace_name, scale_bit, adaptive_bit)
    return mcmc_file, log_file, png_dir
    
    
def hierarchical_mcmc_files(model, expt_name, first_trace_name, num_traces, parallel):
    if arcus_b:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expandvars("projects/PyAP/python/output/")
    if parallel:
        mcmc_dir = first_bit+"{}/hierarchical_mcmc_parallel/{}/{}_traces/model_{}/".format(expt_name, first_trace_name, num_traces, model)
    else:
        mcmc_dir = first_bit+"{}/hierarchical_mcmc_series/{}/{}_traces/model_{}/".format(expt_name, first_trace_name, num_traces, model)
    txt_dir, png_dir, pdf_dir = mcmc_dir+"chain/", mcmc_dir+"figs/png/", mcmc_dir+"figs/pdf/"
    for d in [txt_dir, png_dir, pdf_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    mcmc_file = txt_dir+"{}_hMCMC_{}_with_{}_traces_model_{}.txt".format(expt_name, first_trace_name, num_traces, model)
    log_file = mcmc_dir+"{}_hMCMC_{}_with_{}_traces_model_{}.log".format(expt_name, first_trace_name, num_traces, model)
    return mcmc_file, log_file, png_dir, pdf_dir


def get_original_params(model):
    if (model==1): # Hodgkin Huxley
        model_name = "Hodgkin Huxley 1952"
        original_gs = [120,36,0.3]
        g_parameters = ['G_{Na}', 'G_K', 'G_l']
    elif (model==2): # Beeler Reuter
        model_name = "Beeler Reuter 1977"
        original_gs = [0.04, 0.0035, 0.008, 9e-4]
        g_parameters = ['G_{Na}', 'G_{K1}', 'G_K', 'G_{CaL}']
    elif (model==3): # Luo Rudy
        model_name = "Luo Rudy 1991"
        original_gs = [23, 0.282, 0.6047, 0.09, 0.03921, 0.0183]
        g_parameters = ['G_{Na}', 'G_K', 'G_{K1}','G_{CaL}', 'G_l', 'G_{Kp}']
    elif (model==4): # Ten Tusscher Epi. 04
        model_name = "ten Tusscher 2004 epi."
        original_gs = [14.838, 0.000175, 5.405, 0.096, 0.245, 1000, 0.294, 0.000592, 0.00029, 0.825, 0.0146, 1.362]
        g_parameters = ['G_{Na}', 'G_{CaL}', 'G_{K1}', 'G_{Kr}', 'G_{Ks}', 'k_{NaCa}',
                          'G_{to}', 'G_{bCa}', 'G_{bNa}', 'G_{pCa}', 'G_{pK}', 'P_{NaK}']
    elif (model==5): # O'Hara Rudy Endo. 11
        model_name = "O'Hara 2011 endo."
        original_gs = [75, 0.0001, 0.003, 0.1908, 0.046, 0.0034, 0.0008, 30, 0.02, 2.5e-8, 3.75e-10, 0.0005, 0.0075]
        g_parameters = ['G_{Na}', 'G_{CaL}', 'G_{bK}', 'G_{K1}',
                          'G_{Kr}', 'G_{Ks}', 'k_{NaCa}', 'P_{NaK}',
                          'G_{to}', 'G_{bCa}', 'G_{bNa}', 'G_{pCa}', 'G_{NaL}']
    elif (model==6): # Davies 2012 dog
        model_name = "Davies 2012 canine"
        original_gs = [8.25, 0.000243, 0.5, 0.00276,
                   0.00746925, 0.0138542, 0.0575, 0.0000007980336,
                   5.85, 0.61875, 0.1805, 4e-7,
                   0.000225, 0.011]
        g_parameters = ['G_{Na}', 'G_{CaL}', 'G_{K1}', 'G_{pK}',
                        'G_{Ks}', 'G_{Kr}', 'G_{pCa}', 'G_{bCa}',
                        'k_{NaCa}', 'P_{NaK}', 'G_{to1}', 'G_{to2}',
                        'G_{bCl}', 'G_{NaL}']
    elif (model==7): # Paci ventricular-like SC-CM
        model_name = "Paci ventricular-like SC-CM"
        original_gs = [3671.2302, 8.635702e-5, 28.1492, 2.041,
                   29.8667, 0.4125, 4900, 0.69264,
                   1.841424, 29.9038, 0.9, 30.10312]
        g_parameters = ['G_{Na}', 'G_{CaL}', 'G_{K1}', 'G_{Ks}',
                        'G_{Kr}', 'G_{pCa}', 'K_{NaCa}', 'G_{bCa}',
                        'P_{NaK}', 'G_{to}', 'G_{bNa}', 'G_f']
    elif (model==8): # Gokhale 2017 ex293
        model_name = "Gokhale 2017 ex293"
        original_gs = [90.34, 6.609, 0.6976, 0.1332, 0.9, 0.75]
        g_parameters = ['G_{Na}', 'G_{K1}', 'G_{Na,wt}', 'G_{K1,wt}'
                        'd', 'f']
    original_gs = np.array(original_gs)
    return original_gs, g_parameters, model_name
    
    
def get_protocol_details(protocol): # pre-defined protocols
    if (protocol==1):
        solve_start = 0.
        solve_end = 500.
        solve_timestep = 0.2
        stimulus_magnitude = -25.5
        stimulus_duration = 2
        stimulus_period = 1000
        stimulus_start_time = 20
    return solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time
    
    
def synthetic_nonhierarchical_chain_file_and_figs_dir(model,protocol,python_seed): # synthetic data
    # keeping it outside of Chaste build folder, in case that gets wiped in a clean build, or something
    output_dir = os.path.expanduser('projects/RossJ/python/output/synthetic/nonhierarchical/model_{}/protocol_{}/python_seed_{}/'.format(model,protocol,python_seed))
    chain_dir = output_dir + 'chain/'
    figs_dir = output_dir + 'figures/'
    for d in [chain_dir,figs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    chain_file = chain_dir + 'model_{}_protocol_{}_python_seed_{}_synthetic_nonhierarchical_mcmc.txt'.format(model,protocol,python_seed)
    return chain_file, figs_dir
    
    
def synthetic_hierarchical_chain_file_and_figs_dir(model,protocol,num_expts,python_seed): # synthetic data
    # keeping it outside of Chaste build folder, in case that gets wiped in a clean build, or something
    output_dir = os.path.expanduser('~/RossJ_output/synthetic/hierarchical/model_{}/protocol_{}/num_expts_{}/python_seed_{}/'.format(model,protocol,num_expts,python_seed))
    chain_dir = output_dir + 'chain/'
    figs_dir = output_dir + 'figures/'
    for d in [chain_dir,figs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    chain_file = chain_dir + 'model_{}_protocol_{}_num_expts_{}_python_seed_{}_synthetic_hierarchical_mcmc.txt'.format(model,protocol,num_expts,python_seed)
    info_file = output_dir + 'model_{}_protocol_{}_num_expts_{}_python_seed_{}_synthetic_hierarchical_info.txt'.format(model,protocol,num_expts,python_seed)
    return chain_file, figs_dir, info_file
    

def dog_trace_path(trace_number):
    return "projects/PyAP/python/input/dog_teun_csv/dog_AP_trace_{}.csv".format(trace_number)
    

def roche_trace_path(trace_number):
    return "projects/PyAP/python/input/roche_170123_2_2_csv/Trace_2_2_{}_1.csv".format(trace_number)


def dog_cmaes_path(model_number, trace_number):
    if arcus_b:
        cmaes_dir = os.path.expandvars("$DATA/PyAP_output/dog_teun/cmaes/model_{}/".format(model_number))
    else:
        cmaes_dir = "projects/PyAP/python/output/dog_teun/cmaes/model_{}/".format(model_number)
    if not os.path.exists(cmaes_dir):
        os.makedirs(cmaes_dir)
    return cmaes_dir, cmaes_dir+"dog_model_{}_trace_{}_cmaes_best_fit.txt".format(model_number, trace_number)


def dog_data_clamp_unscaled_mcmc_file(model_number, trace_number):
    if arcus_b:
        mcmc_dir = os.path.expandvars("$DATA/PyAP_output/dog_teun/adaptive_mcmc/unscaled/model_{}/trace_{}/".format(model_number, trace_number))
    else:
        mcmc_dir = "projects/PyAP/python/output/dog_teun/adaptive_mcmc/unscaled/model_{}/trace_{}/".format(model_number, trace_number)
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)
    return mcmc_dir, mcmc_dir+"dog_model_{}_trace_{}_adaptive_mcmc_unscaled.txt".format(model_number, trace_number)


def dog_data_clamp_exp_scaled_mcmc_file(model_number, trace_number):
    if arcus_b:
        mcmc_dir = os.path.expandvars("$DATA/PyAP_output/dog_teun/adaptive_mcmc/exp_scaled/model_{}/trace_{}/".format(model_number, trace_number))
    else:
        mcmc_dir = "projects/PyAP/python/output/dog_teun/adaptive_mcmc/exp_scaled/model_{}/trace_{}/".format(model_number, trace_number)
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)
    return mcmc_dir, mcmc_dir+"dog_model_{}_trace_{}_adaptive_mcmc_exp_scaled.txt".format(model_number, trace_number)


def roche_cmaes_path(model_number, trace_number):
    if arcus_b:
        cmaes_dir = os.path.expandvars("$DATA/PyAP_output/roche_170123_2_2/cmaes/model_{}/".format(model_number))
    else:
        cmaes_dir = "projects/PyAP/python/output/roche_170123_2_2/cmaes/model_{}/".format(model_number)
    if not os.path.exists(cmaes_dir):
        os.makedirs(cmaes_dir)
    return cmaes_dir, cmaes_dir+"roche_170123_2_2_model_{}_trace_{}_cmaes_best_fit.txt".format(model_number, trace_number)


def roche_data_clamp_exp_scaled_mcmc_file(model_number, trace_number):
    if arcus_b:
        mcmc_dir = os.path.expandvars("$DATA/PyAP_output/roche_170123_2_2/adaptive_mcmc/exp_scaled/model_{}/trace_{}/".format(model_number, trace_number))
    else:
        mcmc_dir = "projects/PyAP/python/output/roche_170123_2_2/adaptive_mcmc/exp_scaled/model_{}/trace_{}/".format(model_number, trace_number)
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)
    return mcmc_dir, mcmc_dir+"roche_170123_2_2_model_{}_trace_{}_adaptive_mcmc_exp_scaled.txt".format(model_number, trace_number)









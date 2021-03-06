"""
CHOICE OF CELL MODEL 

# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine) 2012
# 7. Paci (SC-CM ventricular)
# 8. Gokhale 2017 ex293
# 9. Decker (canine) 2009
# 666. BR with null parameter (to test hMCMC)

"""

import numpy as np
import os
import sys
import socket

arcus = False
arcus_b = False
if socket.getfqdn().endswith("arcus.arc.local"):
    arcus_b = True
    print "\nHopefully on arcus-b\n"
elif socket.getfqdn().endswith("arcus.osc.local"):
    arcus = True
    print "\nShould be on arcus\n"


def compute_apd90(times, trace, stim_start_time):
    approx_resting_V = np.mean(trace[times<stim_start_time])
    max_V_diff = np.max(trace) - approx_resting_V
    adp90_threshold = approx_resting_V + 0.1*max_V_diff
    start_clock = np.min(times[trace >= adp90_threshold])
    stop_clock = np.max(times[trace >= adp90_threshold])
    return stop_clock - start_clock


def cmaes_and_figs_files(model_number, expt_name, trace_name, unscaled):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    if unscaled:
        cmaes_dir = first_bit+"{}/cmaes/model_{}/unscaled/".format(expt_name, model_number)
    else:
        cmaes_dir = first_bit+"{}/cmaes/model_{}/exp_scaled/".format(expt_name, model_number)
    txt_dir, png_dir, svg_dir = cmaes_dir+"params/", cmaes_dir+"figs/png/", cmaes_dir+"figs/svg/"
    for d in [txt_dir, png_dir, svg_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    cmaes_best_fits_file = txt_dir+"{}_model_{}_trace_{}_cmaes_best_fits.txt".format(expt_name, model_number, trace_name)
    best_fit_png = png_dir+"{}_model_{}_trace_{}_cmaes_best_fit.png".format(expt_name, model_number, trace_name)
    best_fit_svg = svg_dir+"{}_model_{}_trace_{}_cmaes_best_fit.svg".format(expt_name, model_number, trace_name)
    return cmaes_best_fits_file, best_fit_png, best_fit_svg


def cmaes_and_figs_files_lnG(model_number, expt_name, trace_name):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    cmaes_dir = first_bit+"{}/cmaes/model_{}/".format(expt_name, model_number)
    txt_dir, png_dir, svg_dir = cmaes_dir+"params/", cmaes_dir+"figs/png/", cmaes_dir+"figs/svg/"
    for d in [txt_dir, png_dir, svg_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    cmaes_best_fits_file = txt_dir+"{}_model_{}_trace_{}_cmaes_best_fits.txt".format(expt_name, model_number, trace_name)
    best_fit_png = png_dir+"{}_model_{}_trace_{}_cmaes_best_fit.png".format(expt_name, model_number, trace_name)
    best_fit_svg = svg_dir+"{}_model_{}_trace_{}_cmaes_best_fit.svg".format(expt_name, model_number, trace_name)
    return cmaes_best_fits_file, best_fit_png, best_fit_svg
    

def cmaes_log_likelihood_lnG(model_number, expt_name, trace_name):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    cmaes_dir = first_bit+"{}/cmaes/model_{}/".format(expt_name, model_number)
    txt_dir, png_dir, pdf_dir = cmaes_dir+"params/", cmaes_dir+"figs/png/", cmaes_dir+"figs/pdf/"
    for d in [txt_dir, png_dir, pdf_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    cmaes_best_fits_file = txt_dir+"{}_model_{}_trace_{}_cmaes_best_log_likelihoods.txt".format(expt_name, model_number, trace_name)
    best_fit_png = png_dir+"{}_model_{}_trace_{}_cmaes_best_log_likelihood.png".format(expt_name, model_number, trace_name)
    best_fit_pdf = pdf_dir+"{}_model_{}_trace_{}_cmaes_best_log_likelihood.pdf".format(expt_name, model_number, trace_name)
    return cmaes_best_fits_file, best_fit_png, best_fit_pdf


def mcmc_file_log_file_and_figs_dirs(model_number, expt_name, trace_name, unscaled, non_adaptive, temperature):
    if unscaled:
        scale_bit = "mcmc_unscaled"
    else:
        scale_bit = "mcmc_exp_scaled"
    if non_adaptive:
        adaptive_bit = "non_adaptive"
    else:
        adaptive_bit = "adaptive"
    if arcus_b:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    mcmc_dir = first_bit+"{}/{}/{}/model_{}/{}/temperature_{}/".format(expt_name, scale_bit, adaptive_bit, model_number, trace_name, temperature)
    txt_dir, png_dir = mcmc_dir+"chains/", mcmc_dir+"figs/png/"
    for d in [txt_dir, png_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    mcmc_file = txt_dir+"{}_model_{}_trace_{}_{}_{}_temperature_{}.txt".format(expt_name, model_number, trace_name, scale_bit, adaptive_bit, temperature)
    log_file = mcmc_dir+"{}_model_{}_trace_{}_{}_{}_temperature_{}.log".format(expt_name, model_number, trace_name, scale_bit, adaptive_bit, temperature)
    return mcmc_file, log_file, png_dir
    
    
def mcmc_lnG_file_log_file_and_figs_dirs(model_number, expt_name, trace_name):
    adaptive_bit = "adaptive_mcmc_lnG"
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    mcmc_dir = first_bit+"{}/{}/model_{}/{}/".format(expt_name, adaptive_bit, model_number, trace_name)
    txt_dir, png_dir = mcmc_dir+"chains/", mcmc_dir+"figs/png/"
    for d in [txt_dir, png_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    mcmc_file = txt_dir+"{}_lnG_model_{}_trace_{}_{}.txt".format(expt_name, model_number, trace_name, adaptive_bit)
    log_file = mcmc_dir+"{}_lnG_model_{}_trace_{}_{}.log".format(expt_name, model_number, trace_name, adaptive_bit)
    return mcmc_file, log_file, png_dir
    
    
def hierarchical_mcmc_files(model, expt_name, first_trace_name, num_traces, parallel):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
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


def hierarchical_lnG_mcmc_files(model, expt_name, first_trace_name, num_traces, parallel):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    mcmc_dir = first_bit+"{}/hierarchical_lnG_mcmc_parallel/{}/{}_traces/model_{}/".format(expt_name, first_trace_name, num_traces, model)
    txt_dir, png_dir, pdf_dir = mcmc_dir+"chain/", mcmc_dir+"figs/png/", mcmc_dir+"figs/pdf/"
    for d in [txt_dir, png_dir, pdf_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    mcmc_file = txt_dir+"{}_hMCMC_lnG_{}_with_{}_traces_model_{}.txt".format(expt_name, first_trace_name, num_traces, model)
    log_file = mcmc_dir+"{}_hMCMC_lnG_{}_with_{}_traces_model_{}.log".format(expt_name, first_trace_name, num_traces, model)
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
    elif (model==666): # Beeler Reuter with null parameter
        model_name = "Beeler Reuter 1977 with null parameter"
        original_gs = [0.04, 0.0035, 0.008, 9e-4, (0.0005/75)*0.04]  # last param to be same relative size as in OH GpCa
        g_parameters = ['G_{Na}', 'G_{K1}', 'G_K', 'G_{CaL}', 'G_{null}']
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
                        'G_{Kr}', 'G_{pCa}', 'k_{NaCa}', 'G_{bCa}',
                        'P_{NaK}', 'G_{to}', 'G_{bNa}', 'G_f']
    elif (model==8): # Gokhale 2017 ex293
        model_name = "Gokhale 2017 ex293"
        original_gs = [90.34, 6.609, 0.6976, 0.1332, 0.9, 0.75]
        g_parameters = ['G_{Na}', 'G_{K1}', 'G_{Na,wt}', 'G_{K1,wt}'
                        'd', 'f']
    elif (model==9): # Decker canine 2009
        model_name = "Decker 2009 canine"
        original_gs = [9.075, 0.00015552, 0.5, 0.00276,
                       0.0826, 0.0138542, 0.0575, 1.99508e-7,
                       4.5, 1.4, 0.497458, 9e-7, 0.000225, 0.0065]
        g_parameters = ['G_{Na}', 'G_{CaL}', 'G_{K1}', 'G_{pK}',
                        'G_{Ks}', 'G_{Kr}', 'G_{pCa}', 'G_{bCa}',
                        'k_{NaCa}', 'P_{NaK}', 'G_{to1}', 'G_{to2}',
                        'G_{bCl}', 'G_{NaL}']
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
    

def gary_predictive_file(expt_name, num_traces, param_idx):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    garydir = first_bit + "{}/gary_predictive/{}_traces/".format(expt_name, num_traces)
    garydir_png = garydir + "png/"
    if not os.path.exists(garydir_png):
        os.makedirs(garydir_png)
    return garydir + "{}_{}_traces_gary_predictive_parameter_{}.txt".format(expt_name, num_traces, param_idx), garydir_png
    

def cmaes_final_state_vars_file(model_number, expt_name, trace_name):
    if arcus_b or arcus:
        first_bit = os.path.expandvars("$DATA/PyAP_output/")
    else:
        first_bit = os.path.expanduser("~/PyAP_output/")
    cmaes_dir = first_bit+"{}/cmaes/model_{}/".format(expt_name, model_number)
    final_state_vars_file = cmaes_dir + "{}_model_{}_{}_final_state_vars.txt".format(expt_name, model_number, trace_name)
    return final_state_vars_file






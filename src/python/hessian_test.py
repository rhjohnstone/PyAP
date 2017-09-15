import ap_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import pyap_setup as ps
import sys
import itertools as it
import numpy.random as npr
import numdifftools as nd
import cma


def example_likelihood_function(trace):
    return np.sum(trace**2)
    
    
def sos(test_trace):
    return np.sum((expt_trace-test_trace)**2)
    
    
def get_test_trace(g_params):
    ap.SetToModelInitialConditions()
    return ap.SolveForVoltageTraceWithParams(g_params)
    

def exponential_scaling(unscaled_params):
    return original_gs ** (unscaled_params/10.)
    
    
def obj(unscaled_Gs):
    return np.sum((expt_trace-get_test_trace(exponential_scaling(unscaled_Gs)))**2)



        
def log_likelihood(_params):
    global count
    count += 1
    #print _params
    if np.any(_params<0):
        print _params
        return -np.inf
    else:
        Gs, sigma = _params[:-1], _params[-1]
        #print Gs
        test_trace = get_test_trace(Gs)
        return -num_pts*np.log(sigma) - np.sum((expt_trace-test_trace)**2)/(2.*sigma**2)


# 1. Hodgkin Huxley
# 2. Beeler Reuter
# 3. Luo Rudy
# 4. ten Tusscher
# 5. O'Hara Rudy
# 6. Davies (canine)
# 7. Paci (SC-CM ventricular) (not available atm)
# 8. Gokhale 2017 ex293 (not available atm)

protocol = 1
seed = 1
npr.seed(seed)
noise_sd = 0.25


solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)
for model_number in xrange(5,6):

    solve_end = 100
    times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

    original_gs, g_parameters, model_name = ps.get_original_params(model_number)
    original_gs = np.array(original_gs)

    ap = ap_simulator.APSimulator()
    ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
    ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
    ap.DefineModel(model_number)
    
    ap.SetToModelInitialConditions()
    expt_trace = ap.SolveForVoltageTraceWithParams(original_gs)
    num_pts = len(expt_trace)
    expt_trace += noise_sd*npr.randn(num_pts)
    print "expt_trace original_gs done"
    
    sigma = 0.25
    true_params = np.concatenate((original_gs, [noise_sd]))
    
    x0 = 10.*np.ones(len(original_gs))
    sigma0 = 0.1
    print "x0:", x0
    print "sigma0:", sigma0
    obj0 = obj(x0)
    print "obj0:", round(obj0, 2)
    es = cma.CMAEvolutionStrategy(x0, sigma0)#, options)
    while not es.stop():
        X = es.ask()
        es.tell(X, [obj(x) for x in X])
        es.disp()
    res = es.result()
    best_gs = exponential_scaling(res[0])
    print "best_gs:", best_gs
    
    best_sigma = np.sqrt(res[1]/len(expt_trace))
    print "best_sigma:", best_sigma
    
    best_params = np.concatenate((best_gs,[best_sigma]))
    
    """count = 0
    hess = nd.Hessian(log_likelihood, step=0.0001*true_params, step_ratio=1, num_steps=50)(true_params)
    print "\n\n", hess, "\n\n"
    print "count:", count"""
    
    count = 0
    hess = nd.Hessian(log_likelihood, step=0.00001*best_params, step_ratio=1.1, num_steps=100)(best_params)
    print "\n\n", hess, "\n\n"
    print "count:", count
    
    fisher = -hess
    sigma = np.sqrt(np.diag(np.linalg.pinv(fisher)))
    upper = true_params + 1.96*sigma
    lower = true_params - 1.96*sigma
    
    print "\n", np.vstack((lower,upper)).T, "\n"
    
    
    
    
    
    
    

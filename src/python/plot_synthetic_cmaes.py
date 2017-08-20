import pyap_setup as ps
import ap_simulator
import numpy as np
import numpy.random as npr
import argparse
import matplotlib.pyplot as plt
import sys
import cma

seed = 1
npr.seed(seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--model", type=int, help="AP model number", required=True)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

protocol = 1
solve_start, solve_end, solve_timestep, stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time = ps.get_protocol_details(protocol)
if args.model == 1:
    solve_end = 100  # just for HH
original_gs, g_parameters, model_name = ps.get_original_params(args.model)
num_gs = len(original_gs)
num_solves = 1

expt_params = np.copy(original_gs)


def exponential_scaling(unscaled_params):
    return original_gs ** (unscaled_params/10.)


def solve_for_voltage_trace(temp_g_params, _ap_model):
    _ap_model.SetToModelInitialConditions()
    try:
        return _ap_model.SolveForVoltageTraceWithParams(temp_g_params)
    except ap_simulator.CPPException, e:
        print e.GetShortMessage
        sys.exit()
    
    
def obj(temp_test_params, temp_ap_model):
    scaled_params = exponential_scaling(temp_test_params)
    temp_test_trace = solve_for_voltage_trace(scaled_params, temp_ap_model)
    return np.sum((temp_test_trace-expt_trace)**2)


ap_model = ap_simulator.APSimulator()
ap_model.DefineStimulus(stimulus_magnitude, stimulus_duration, stimulus_period, stimulus_start_time)
ap_model.DefineModel(args.model)
ap_model.DefineSolveTimes(solve_start, solve_end, solve_timestep)
ap_model.SetNumberOfSolves(num_solves)

expt_times = np.arange(solve_start, solve_end+solve_timestep, solve_timestep)
expt_trace = solve_for_voltage_trace(expt_params, ap_model) + 0.25*npr.randn(len(expt_times))

fig, axs = plt.subplots(2, 3, sharey=True)
#ax = fig.add_subplot(111)
#ax.grid()
#ax.set_xlabel('Time (ms)')
#ax.set_ylabel('Membrane voltage (mV)')
#ax.plot(expt_times, expt_trace, label="Expt")

x0 = 10. + 5.*npr.randn(num_gs)
x0[x0<0] = 0.
print "x0:", x0
obj0 = obj(x0, ap_model)
print "obj0:", round(obj0, 2)
sigma0 = 0.1
es = cma.CMAEvolutionStrategy(x0, sigma0)#, options)
it = 0
test_its = [0, 10, 100]
axi = 0
while not es.stop():
    if it in test_its:
        temp_gs = exponential_scaling(es.mean)
        temp_percents = temp_gs / original_gs
        axs[1, axi].grid()
        axs[1, axi].set_xlabel('Time (ms)')
        axs[1, axi].plot(expt_times, expt_trace, label="Expt")
        axs[1, axi].plot(expt_times, solve_for_voltage_trace(temp_gs, ap_model), label="Iteration {}".format(it))
        axs[1, axi].legend()
        #axs[0,axi].bar(temp_percents)
        axi += 1
    X = es.ask()
    es.tell(X, [obj(x, ap_model) for x in X])
    es.disp()
    it += 1
print "{} iterations total".format(it)
res = es.result()

best_gs = res[0]
#ax.plot(expt_times, solve_for_voltage_trace(exponential_scaling(best_gs), ap_model), label="Best fit")

axs[1,0].set_ylabel('Membrane voltage (mV)')
fig.tight_layout()
plt.show(block=True)












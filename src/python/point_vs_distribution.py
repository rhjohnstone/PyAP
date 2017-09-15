import ap_simulator
import pyap_setup as ps
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)


def dose_response_model(dose,hill,IC50):
    return 100. * ( 1. - 1./(1.+(1.*dose/IC50)**hill) )


crumb_file = "projects/PyAP/python/crumb_data.csv"
df = pd.read_csv(crumb_file, names=['Drug','Channel','Experiment','Concentration','Inhibition'],skiprows=1)

drug = "Amiodarone"
channel = "hERG"

experiment_numbers = np.array(df[(df['Drug'] == drug) & (df['Channel'] == channel)].Experiment.unique())
num_expts = max(experiment_numbers)
experiments = []
for expt in experiment_numbers:
    experiments.append(np.array(df[(df['Drug'] == drug) & (df['Channel'] == channel) & (df['Experiment'] == expt)][['Concentration','Inhibition']]))
experiment_numbers -= 1

concs = np.array([])
responses = np.array([])
for i in xrange(num_expts):
    concs = np.concatenate((concs,experiments[i][:,0]))
    responses = np.concatenate((responses,experiments[i][:,1]))

best_hill, best_pic50 = 5.824205286421374472e-01, 6.026421289700708783e+00

chain = np.loadtxt("projects/PyAP/python/crumb_data_Amiodarone_hERG_nonhierarchical_vary_hill_chain.txt", usecols=range(2))

total_its, num_params = chain.shape
burn = total_its/4

num_samples = 250
samples = chain[npr.randint(burn, total_its, num_samples), :]


protocol = 1

solve_start,solve_end,solve_timestep,stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time = ps.get_protocol_details(protocol)
times = np.arange(solve_start,solve_end+solve_timestep,solve_timestep)

model_number = 5
original_gs, g_parameters, model_name = ps.get_original_params(model_number)
original_gs = np.array(original_gs)

conc = 0.5

ap = ap_simulator.APSimulator()
ap.DefineStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time)
ap.DefineSolveTimes(solve_start,solve_end,solve_timestep)
ap.DefineModel(model_number)


best_fit_fig = plt.figure()#figsize=(5,4))

best_fit_ax = best_fit_fig.add_subplot(221)
best_fit_ax.set_title('Ion channel screening')
best_fit_ax.set_xscale('log')
best_fit_ax.grid()
plot_lower_lim = int(np.log10(np.min(concs)))-1
plot_upper_lim = int(np.log10(np.max(concs)))+2
best_fit_ax.set_xlim(10**plot_lower_lim, 10**plot_upper_lim)
best_fit_ax.set_ylim(0, 100)
num_pts = 1001
x_range = np.logspace(plot_lower_lim, plot_upper_lim, num_pts)
best_fit_curve = dose_response_model(x_range, best_hill, pic50_to_ic50(best_pic50))
best_fit_ax.plot(x_range,best_fit_curve,label='Best fit',lw=2)
best_fit_ax.set_ylabel('% {} block'.format(channel))
#best_fit_ax.set_title('Hill = {}, pIC50 = {}'.format(np.round(hill_cur,2),np.round(pic50_cur,2)))
best_fit_ax.scatter(concs,responses,marker="o",color='orange',s=100,label='Data',zorder=10)
best_fit_ax.legend(loc=2)
best_fit_ax.tick_params(labelbottom='off')
best_fit_ax.axvline(conc, color='green', lw=2)

best_AP_ax = best_fit_fig.add_subplot(222)
best_AP_ax.set_title("{} predictions".format(model_name))
best_AP_ax.grid()
ap.SetToModelInitialConditions()
temp_gs = np.copy(original_gs)
scale = dose_response_model(conc, best_hill, pic50_to_ic50(best_pic50))/100.
print scale
temp_gs[4] *= (1.-scale)
trace = ap.SolveForVoltageTraceWithParams(temp_gs)
best_AP_ax.plot(times, trace, lw=2, color='blue')
best_AP_ax.set_ylabel('Membrane voltage (mV)')
best_AP_ax.legend(loc=2)
best_AP_ax.tick_params(labelbottom='off')



samples_ax = best_fit_fig.add_subplot(223, sharex=best_fit_ax)
samples_ax.set_xscale('log')
samples_ax.grid()
plot_lower_lim = int(np.log10(np.min(concs)))-1
plot_upper_lim = int(np.log10(np.max(concs)))+2
samples_ax.set_xlim(10**plot_lower_lim, 10**plot_upper_lim)
samples_ax.set_ylim(0, 100)
x_range = np.logspace(plot_lower_lim, plot_upper_lim, num_pts)
for i in xrange(num_samples):
    if i==0:
        samples_ax.plot(x_range, dose_response_model(x_range, samples[i, 0], pic50_to_ic50(samples[i, 1])),lw=2, alpha=0.05, color='blue', label="Fits")
    else:
        samples_ax.plot(x_range, dose_response_model(x_range, samples[i, 0], pic50_to_ic50(samples[i, 1])),lw=2, alpha=0.05, color='blue')
    
samples_ax.set_ylabel('% {} block'.format(channel))
samples_ax.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
#samples_ax.set_title('Hill = {}, pIC50 = {}'.format(np.round(hill_cur,2),np.round(pic50_cur,2)))
samples_ax.scatter(concs,responses,marker="o",color='orange',s=100,label='Data',zorder=10)
samples_ax.legend(loc=2)
samples_ax.axvline(conc, color='green', lw=2)

samples_AP_ax = best_fit_fig.add_subplot(224, sharex=best_AP_ax)
samples_AP_ax.grid()
for i in xrange(num_samples):
    ap.SetToModelInitialConditions()
    temp_gs = np.copy(original_gs)
    hill, pic50 = samples[i, :]
    scale = dose_response_model(conc, hill, pic50_to_ic50(pic50))/100.
    temp_gs[4] *= (1.-scale)
    trace = ap.SolveForVoltageTraceWithParams(temp_gs)
    samples_AP_ax.plot(times, trace, lw=2, color='blue', alpha=0.05)
samples_AP_ax.set_ylabel('Membrane voltage (mV)')
samples_AP_ax.set_xlabel("Time (ms)")
#samples_ax.set_title('Hill = {}, pIC50 = {}'.format(np.round(hill_cur,2),np.round(pic50_cur,2)))

arrow = "<|-, head_length=.8, head_width=.8"#, tail_width=.4"

con = ConnectionPatch(xyA=(100,-30), xyB=(10,50), coordsA="data", coordsB="data", arrowstyle=arrow, axesA=best_AP_ax, axesB=best_fit_ax, color="red", lw=4)
best_AP_ax.add_artist(con)

con2 = ConnectionPatch(xyA=(100,-30), xyB=(10,50), coordsA="data", coordsB="data", arrowstyle=arrow, axesA=samples_AP_ax, axesB=samples_ax, color="red", lw=4)
samples_AP_ax.add_artist(con2)



best_fit_fig.tight_layout()
best_fit_fig.savefig('point_vs_distn.png')
best_fit_fig.savefig('point_vs_distn.pdf')
#best_fit_fig.savefig(images_dir+'{}_{}_CMA-ES_best_fit.png'.format(drug,channel))
plt.show(block=True)


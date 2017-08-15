import ap_simulator  
import pyap_setup as ps
import argparse
import numpy as np
import sys
import numpy.random as npr
import time
import multiprocessing as mp
import argparse

python_seed = 1
npr.seed(python_seed)

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="first csv file from which to read in data", required=True)
requiredNamed.add_argument("--num-traces", type=int, help="number of traces to fit to, including the one specified as argument", required=True)
parser.add_argument("-i", "--iterations", type=int, help="total MCMC iterations", default=500000)
#parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
args, unknown = parser.parse_known_args()
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
args, unknown = parser.parse_known_args()
trace_path = args.data_file
split_trace_path = trace_path.split('/')
expt_name = split_trace_path[4]
trace_name = split_trace_path[-1][:-4]
options_file = '/'.join( split_trace_path[:5] ) + "/PyAP_options.txt"

def trace_label(dog):
    if (dog < 10):
        return "00"+str(dog)
    elif (10 <= dog < 100):
        return "0"+str(dog)
    else:
        return str(dog)

#num_processors = multiprocessing.cpu_count()
#num_processes = min(num_processors-1,N_e) # counts available cores and makes one fewer process
    
expt_noise = 0.25

num_params, true_gs, g_parameters, g_intervals = ModelSetup.ChooseModel(model) # don't need g_intervals
true_gs = true_gs[:-1]
num_params -= 1 # sigma is included in the original num_params
ModelSetup.CheckProtocolChoice(protocol)

# defined in generate_synthetic_data.py, should abstract(?) this so it's definitely consistent
scale_for_generating_expt_params = 0.1
top_theta = true_gs
top_sigma_squared = (scale_for_generating_expt_params * true_gs)**2

trace_directory = "../input/dog/dog_traces_csv/"


cmaes_directory = "../output/dog/m"+str(model)+"/p"+str(protocol)+"/best_fits/"
params_files = [cmaes_directory+"trace_"+trace_label(i)+"_cmaes_output.txt" for i in range(first_dog,first_dog+N_e)]
start = time.time()
starting_points = np.array([params[np.argmax(params[:,-1]),:-1] for params in [np.loadtxt(cmaes_file) for cmaes_file in params_files]])

#sys.exit()


starting_mean = np.mean(starting_points,axis=0)
if N_e == 1:
    starting_vars = 0.1*np.ones(len(starting_mean))
else:
    starting_vars = np.var(starting_points,axis=0)
    
print "starting_mean:\n",starting_mean
    
print "starting_points:\n", starting_points

def chain_segment_file(chain_dir,model,protocol,N_e,segment):
    if (0 <= segment < 10):
        seg_str = "00"+str(segment)
    elif (10 <= segment < 100):
        seg_str = "0"+str(segment)
    elif (100 <= segment < 1000):
        seg_str = str(segment)
    return chain_dir + "m_"+str(model)+"_p_"+str(protocol)+"_N_e_"+str(N_e)+"_hierarchical_"+seg_str+".txt"


directory = "../output/hierarchical/dog_2/m"+str(model)+"/p"+str(protocol)+"/post_cmaes/first_dog_"+str(first_dog)+"/"+str(N_e)+"_expts/"
if not os.path.exists(directory):
    os.makedirs(directory)
chain_dir = directory + "chain/"
if not os.path.exists(chain_dir):
    os.makedirs(chain_dir)
else:
    for the_file in os.listdir(chain_dir):
        file_path = os.path.join(chain_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e
    
gif_dir = directory+'gif/'
    
segment = 0
output_file = chain_segment_file(chain_dir,model,protocol,N_e,segment)
with open(output_file,'w') as outfile:
    pass

print "directory:\n",directory

old_eta_js = np.zeros((num_params,4))
old_eta_js[:,0] = starting_mean
old_eta_js[:,1] = 1. * N_e
old_eta_js[:,2] = 0.5 * N_e
old_eta_js[:,3] = 0.5 * N_e * starting_vars
#old_eta_js[:,3] = starting_vars * (old_eta_js[:,2]+1)

print "old_eta_js:\n", old_eta_js



cells = []
expt_traces = []
for i in range(N_e):
    trace = first_dog + i
    cells.append(cs.CellSimulator(model, protocol, c_seed, 1, true_gs, expt_noise, trace))
    expt_traces.append(cells[i].get_expt_traces())
expt_traces = np.array(expt_traces)


times = cells[0].get_expt_times()
num_pts = len(times)

print "\n\n***\nCreated cell and got expt_traces and times\n***\n\n"
print "times:\n", times
print "expt_traces:\n", expt_traces



#sys.exit()


def get_test_trace(params,index):
    return cells[index].solve_for_voltage_with_params(params)

#print "old_eta_js =\n", old_eta_js

uniform_noise_prior = [0.,100.*expt_noise]

def new_eta(old_eta,samples): # for sampling from conjugate prior-ed N-IG
    assert(len(old_eta)==4)
    x_bar = np.mean(samples)
    num_samples = len(samples)
    mu, nu, alpha, beta = 1.*old_eta
    new_mu = ((nu*mu + num_samples*x_bar) / (nu + num_samples))
    new_nu = nu + num_samples
    new_alpha = alpha + 0.5*num_samples
    new_beta = beta + 0.5*np.sum((samples-x_bar)**2) + 0.5*((num_samples*nu)/(nu+num_samples))*(x_bar-mu)**2
    return new_mu,new_nu,new_alpha,new_beta
    
def sample_from_N_IG(eta):
    mu, nu, alpha, beta = eta
    sigma_squared = st.invgamma.rvs(alpha,scale=beta)
    sample = mu + np.sqrt(sigma_squared/nu)*npr.randn()
    return sample,sigma_squared
    
def log_pi_theta_i(theta_i,theta,sigma_squareds,sigma,data_i,test_i):
    sum_1 = np.sum(((data_i-test_i)/sigma)**2)
    sum_2 = np.sum(((theta_i-theta)**2)/sigma_squareds)
    return -0.5 * (sum_1 + sum_2)
    
def log_pi_sigma(expt_datas,test_datas,sigma,N_e,num_pts):
    #print expt_datas
    #print test_datas
    if (not (uniform_noise_prior[0] < sigma < uniform_noise_prior[1])):
        return -np.inf
    else:
        return -N_e*num_pts*np.log(sigma) - np.sum((expt_datas-test_datas)**2) / (2*sigma**2)
        
def compute_initial_sigma(expt_datas,test_datas,N_e,num_pts):
    return np.sqrt(np.sum((expt_datas-test_datas)**2) / (N_e*num_pts))
        
    
top_theta_cur = np.copy(starting_mean)
top_sigma_squareds_cur = np.copy(starting_vars)
theta_is_cur = np.copy(starting_points)
#print "theta_is_cur:\n", theta_is_cur


cov_proposal_scale = 0.0000001
sigma_proposal_scale = 0.01

print "top_theta_cur:\n"
print top_theta_cur, "\n"

for i in range(len(top_theta_cur)):
    if (top_theta_cur[i] < 0):
        top_theta_cur[i] = 0
print "top_theta_cur:\n"
print top_theta_cur, "\n"

print "theta_is_cur:\n"
print theta_is_cur, "\n"
        
for row in theta_is_cur:
    for i in range(len(row)):
        if (row[i] < 0):
            row[i] = 0
            
print "theta_is_cur:\n"
print theta_is_cur, "\n"



temp_test_traces_cur = []
for i in range(N_e):
    temp_test_traces_cur.append(get_test_trace(theta_is_cur[i],i))
print temp_test_traces_cur

noise_sigma_cur = compute_initial_sigma(expt_traces,temp_test_traces_cur,N_e,num_pts)

print "noise_sigma_cur:\n", noise_sigma_cur

MCMC = [np.concatenate((top_theta_cur,top_sigma_squareds_cur,theta_is_cur.flatten(),[noise_sigma_cur]))]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for i in range(N_e):
    ax.plot(times,temp_test_traces_cur[i],label='Best fit '+str(first_dog+i))
    ax.plot(times,expt_traces[i],label='Dog '+str(first_dog+i))
ax.set_title('Model '+str(model)+', first dog '+str(first_dog)+', N_e '+str(N_e))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane voltage (mV)')
ax.legend()
fig.tight_layout()
fig.savefig(directory+str(N_e)+'_synthetic_data.png')


#sys.exit()

thinning = 5
MCMC_iterations = 1000000

#pool = multiprocessing.Pool(num_processes) # not sure where best to have this line

covariances = []
for i in range(N_e):
    covariances.append(cov_proposal_scale*np.diag(theta_is_cur[i]))
print "covariances:\n", covariances, "\n"

means = np.copy(theta_is_cur)
print "means:\n", means, "\n"

logas = [0.]*N_e
acceptances = [0.]*N_e
sigma_loga = 0.
sigma_acceptance = 0.

#if t > 1000*number_of_parameters:
def update_covariance_matrix(t,thetaCur,mean_estimate,cov_estimate,loga,accepted):
    s = t - 200*num_params
    gamma_s = 1/(s+1)**0.6
    temp_covariance_bit = np.array([thetaCur-mean_estimate])
    new_cov_estimate = (1-gamma_s) * cov_estimate + gamma_s * np.dot(np.transpose(temp_covariance_bit),temp_covariance_bit)
    new_mean_estimate = (1-gamma_s) * mean_estimate + gamma_s * thetaCur
    new_loga = loga + gamma_s*(accepted-0.25)
    return new_cov_estimate, new_mean_estimate, new_loga
    
"""for alpha in range(1,10):
    beta = 1*(alpha+1)
    gif_fig = plt.figure()
    gif_ax = gif_fig.add_subplot(111)
    gif_range = np.linspace(0,10,1000)
    print "old_eta_js:\n", old_eta_js
    print "old_eta_js[0,2] =", old_eta_js[0,2]
    print "old_eta_js[0,3] =", old_eta_js[0,3]
    #rv = st.invgamma(old_eta_js[0,3]+2,scale=old_eta_js[0,3]+1)
    rv = st.invgamma(alpha,scale=beta)
    print (old_eta_js[0,3]+1) / (old_eta_js[0,3]+1)
    gif_ax.plot(gif_range,rv.pdf(gif_range))
    gif_fig.tight_layout()
    plt.show(block=True)
sys.exit()"""

targets_cur = np.zeros(N_e)
start = time.time()
t = 0
print "About to start MCMC\n"
while True:
    try:
        if ( ( t>0 ) and ( t%500==0 ) ):
            print t, "iterations"
            print "logas =", logas
            print "acceptances =", acceptances
            print "sigma_loga =", sigma_loga
            print "sigma_acceptance =", sigma_acceptance
            segment_size = os.stat(output_file).st_size
            if (segment_size > 100000000):
                segment += 1
                output_file = chain_segment_file(chain_dir,model,protocol,N_e,segment)
                with open(output_file,'w') as outfile:
                    pass
            with open(output_file,'a') as outfile:
                np.savetxt(outfile,MCMC)
            print "sys.getsizeof(MCMC) =", sys.getsizeof(MCMC)
            MCMC = []
        for j in range(num_params):
            temp_eta = new_eta(old_eta_js[j],theta_is_cur[:,j])
            """if (t==3000):
                print "\n\nold_eta_js["+str(j)+"]:\n", old_eta_js[j],"\n"
                print "theta_is_cur[:,"+str(j)+"]:\n", theta_is_cur[:,j],"\n"
                print "temp_eta:\n", temp_eta,"\n"
                print "old_eta_js:\n", old_eta_js, "\n"
                gif_fig = plt.figure()
                gif_ax = gif_fig.add_subplot(111)
                gif_range = np.linspace(0,0.001,1000)
                #rv = st.invgamma(old_eta_js[0,3]+2,scale=old_eta_js[0,3]+1)
                rv = st.invgamma(temp_eta[2],scale=temp_eta[3])
                gif_ax.plot(gif_range,rv.pdf(gif_range))
                gif_fig.tight_layout()
                plt.show(block=True)"""
            while True:
                temp_top_theta_cur, temp_top_sigma_squared_cur = sample_from_N_IG(temp_eta)
                if (temp_top_theta_cur > 0):
                    break
            if (t==3000):
                print "temp_top_theta_cur =", temp_top_theta_cur
                print "temp_top_sigma_squared_cur =", temp_top_sigma_squared_cur
            top_theta_cur[j] = temp_top_theta_cur
            top_sigma_squareds_cur[j] = temp_top_sigma_squared_cur
                    

        # theta i's for each experiment
        
        for i in range(N_e):
            while True:
                theta_i_star = npr.multivariate_normal(theta_is_cur[i],np.exp(logas[i])*covariances[i])
                if (np.all(theta_i_star>=0)):
                    break
        #temp_test_traces_star = pool.map(get_test_trace, theta_is_star)
        #temp_test_traces_star = [get_test_trace(xy) for xy in theta_is_star]
            #if (np.any(theta_i_star<0)):
                #print theta_i_star, "WTF"
            temp_test_trace_star = get_test_trace(theta_i_star,i)
        
            targets_cur[i] = log_pi_theta_i(theta_is_cur[i],top_theta_cur,top_sigma_squareds_cur,noise_sigma_cur,expt_traces[i],temp_test_traces_cur[i])
            target_star = log_pi_theta_i(      theta_i_star,top_theta_cur,top_sigma_squareds_cur,noise_sigma_cur,expt_traces[i],temp_test_trace_star)
            if (3000 <= t < 3020):
                print "t =", t
                print "theta_is_cur["+str(i)+"]:\n", theta_is_cur[i]
                print "theta_i_star:\n", theta_i_star
                print "theta_is_cur["+str(i)+"] - theta_i_star:\n", theta_is_cur[i]-theta_i_star
                print "targets_cur["+str(i)+"]:\n", targets_cur[i]
                print "target_star:\n", target_star
                print "target_star - targets_cur["+str(i)+"] =", target_star - targets_cur[i], "\n"
                print "np.sum((temp_test_traces_cur["+str(i)+"]-temp_test_trace_star)**2) =", np.sum((temp_test_traces_cur[i]-temp_test_trace_star)**2)
                #plt.plot(times,temp_test_traces_cur[i],label='temp_test_traces_cur['+str(i)+']')
                #plt.plot(times,temp_test_trace_star,label='temp_test_trace_star')
                #plt.plot(times,expt_traces[i],label='expt_traces['+str(i)+']')
                #plt.legend()
                #plt.show(block=True)
            u = npr.rand()
            if (np.log(u) < target_star - targets_cur[i]):
                theta_is_cur[i] = np.copy(theta_i_star)
                temp_test_traces_cur[i] = np.copy(temp_test_trace_star)
                accepted = 1
                if (3000 <= t < 3020):
                    print "accepted!\n"
            else:
                accepted = 0
            if (t > 200*num_params):
                temp_cov, temp_mean, temp_loga = update_covariance_matrix(t,theta_is_cur[i],means[i],covariances[i],logas[i],accepted)
                covariances[i] = np.copy(temp_cov)
                means[i] = np.copy(temp_mean)
                logas[i] = temp_loga
            acceptances[i] = (t*acceptances[i] + accepted)/(t+1.)
        # noise sigma
        while True:
            noise_sigma_star = noise_sigma_cur + np.exp(sigma_loga)*sigma_proposal_scale*npr.randn()
            if (uniform_noise_prior[0] < noise_sigma_star < uniform_noise_prior[1]):
                break
        sigma_target_star = log_pi_sigma(expt_traces,temp_test_traces_cur,noise_sigma_star,N_e,num_pts)
        sigma_target_cur = log_pi_sigma(expt_traces,temp_test_traces_cur,noise_sigma_cur,N_e,num_pts)
        u = npr.rand()
        if (np.log(u) < sigma_target_star - sigma_target_cur):
            noise_sigma_cur = noise_sigma_star
            accepted = 1
        else:
            accepted = 0
        sigma_acceptance = (t*sigma_acceptance + accepted)/(t+1)
        if (t > 200*num_params):
            r = t - 200*num_params
            gamma_r = 1/(r+1)**0.6
            sigma_loga += gamma_r*(accepted-0.25)
        if ( t%thinning == 0 ):   
            MCMC.append(np.concatenate((top_theta_cur,top_sigma_squareds_cur,theta_is_cur.flatten(),[noise_sigma_cur])))
        t += 1
    except KeyboardInterrupt:
        print "\nInterrupting MCMC loop"
        #sys.exit()
        break
print "Time taken:", time.time()-start, "s"
#print final_state

print "\nAll done.\n"

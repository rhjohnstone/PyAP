import pyap_setup as ps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--unscaled", action="store_true", help="perform MCMC sampling in unscaled 'conductance space'", default=False)
parser.add_argument("--non-adaptive", action="store_true", help="do not adapt proposal covariance matrix", default=False)
args, unknown = parser.parse_known_args()

model_number = 1
expt_name = "synthetic_HH"
trace_name = "synthetic_HH"

mcmc_file, log_file, png_dir = ps.mcmc_file_log_file_and_figs_dirs(model_number, expt_name, trace_name, args.unscaled, args.non_adaptive)

cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(model_number, expt_name, trace_name)

all_points = np.loadtxt(cmaes_best_fits_file)

data = all_points[:,:-1]

mu = data.mean(axis=0)
centred_data = data - mu
eigenvectors, eigenvalues, V = np.linalg.svd(centred_data.T, full_matrices=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*data.T)
ax.set_title("Hodgkin Huxley 1952 CMA-ES best fits")
ax.set_xlabel('$G_{Na}$')
ax.set_ylabel('$G_K$')
ax.set_zlabel('$G_l$')

scale = 0.1
main_eigv = eigenvectors[:,0]
start = mu - scale*main_eigv
end = mu + scale*main_eigv

print mu
print "eigv:", main_eigv
print "unnormed eigv:", main_eigv/np.linalg.norm(main_eigv)

eigvend = mu + 0.01*eigenvectors[:,0]
line = np.vstack((start, end))
ax.plot(*line.T, color='red', label='First principal component')

#ax.legend()
fig.savefig(png_dir+"cmaes_best_fits.png")
fig.tight_layout()
fig.savefig(png_dir+"cmaes_best_fits_tight.png")
plt.close()

chain = np.loadtxt(mcmc_file, usecols=range(3))
saved_its, num_gs = chain.shape
burn = saved_its/4

num_pts = 500
sample_indices = np.random.randint(burn, saved_its, num_pts)
samples = chain[sample_indices, :]

mu = samples.mean(axis=0)
centred_data = samples - mu
eigenvectors, eigenvalues, V = np.linalg.svd(centred_data.T, full_matrices=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*samples.T)
ax.set_title("Hodgkin Huxley 1952 MCMC output samples")
ax.set_xlabel('$G_{Na}$')
ax.set_ylabel('$G_K$')
ax.set_zlabel('$G_l$')

scale = 0.1
main_eigv = eigenvectors[:,0]
start = mu - scale*main_eigv
end = mu + scale*main_eigv

print mu
print "eigv:", main_eigv
print "unnormed eigv:", main_eigv/np.linalg.norm(main_eigv)

eigvend = mu + 0.01*eigenvectors[:,0]
line = np.vstack((start, end))
ax.plot(*line.T, color='red', label='First principal component')

#ax.legend()
fig.savefig(png_dir+"mcmc_samples.png")
fig.tight_layout()
fig.savefig(png_dir+"mcmc_samples_tight.png")
plt.close()

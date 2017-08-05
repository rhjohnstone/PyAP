import pyap_setup as ps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import PCA

model_number = 1
expt_name = "synthetic_HH"
trace_name = "synthetic_HH"

cmaes_best_fits_file, best_fit_png, best_fit_svg = ps.cmaes_and_figs_files(model_number, expt_name, trace_name)

all_points = np.loadtxt(cmaes_best_fits_file)

data = all_points[:,:-1]

mu = data.mean(axis=0)
centred_data = data - mu
eigenvectors, eigenvalues, V = np.linalg.svd(centred_data.T, full_matrices=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in xrange(3)
    eigvend = mu + 0.5*eigenvectors[i]
    lines = np.vstack((mu, eigvend))
    ax.plot(*lines.T)

ax.scatter(*data.T)
ax.set_title("Hodgkin Huxley 1952 CMA-ES best fits")
ax.set_xlabel('$G_{Na}$')
ax.set_ylabel('$G_K$')
ax.set_zlabel('$G_l$')


plt.show()


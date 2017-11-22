import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, lognorm
import pyap_setup as ps
import sys

def f(v):
    return v**4 - v**3 - 16


fs = 16
phi = 1.61803398875
fig_y = 5

model = 5
original_gs, g_parameters, model_name = ps.get_original_params(model)
num_gs = len(original_gs)

T = 10000

lamb = 4.
alpha = 10.
#beta = 2

coeffs = [1, -1, 0, 0, -16]
roots = np.roots(coeffs)
v = float(roots[0])  # manually ascertained

beta = (alpha-1.)*np.log(v)


num_pts = 201

for i in xrange(num_gs):
    print "{} / {}".format(i+1, num_gs)
    s = np.log(original_gs[i]) + beta / (alpha-1.)
    print "s =", s

    fig = plt.figure(figsize=(phi*fig_y,fig_y))

    ax1 = fig.add_subplot(221)
    ax1.grid()
    offset = 3
    x = np.linspace(int(s)-offset, int(s)+offset, num_pts)
    ax1.plot(x, norm.pdf(x, loc=s, scale=1./np.sqrt(lamb)), lw=2)
    ax1.set_xlabel(r"$\mu$", fontsize=fs)
    ax1.set_ylabel("Prior pdf", fontsize=fs)

    ax2 = fig.add_subplot(222)
    ax2.grid()
    x = np.linspace(0, 4, num_pts)
    ax2.plot(x, gamma.pdf(x, alpha, scale=1./beta), lw=2)
    ax2.set_xlabel(r"$\tau$", fontsize=fs)
    ax2.set_ylabel("Prior pdf", fontsize=fs)

    xmin = int(np.log10(original_gs[i]))-2
    xmax = xmin+4
    x = np.logspace(xmin, xmax, num_pts)
    y = np.zeros(num_pts)
    for T in xrange(T):
        mu = norm.rvs(loc=s, scale=1./np.sqrt(lamb))
        tau = gamma.rvs(alpha, scale=1./beta)
        y += lognorm.pdf(x, s=1./np.sqrt(tau), scale=np.exp(mu))
    y /= T
    ax3 = fig.add_subplot(212)
    ax3.grid()
    ax3.set_xscale('log')
    ax3.plot(x, y, lw=2)
    ax3.axvline(original_gs[i], lw=2, color='red', label='model')
    ax3.set_xlabel(r"${}$".format(g_parameters[i]), fontsize=fs)
    ax3.set_ylabel("Prior pred. pdf", fontsize=fs-2)
    ax3.legend(loc=2, fontsize=fs-2)
    fig.tight_layout()
    fig.savefig("/home/ross/Documents/dphil/chapters/inferring-conductances/ohara_prior_preds/ohara_prior_predictive_{}.png".format(g_parameters[i]))
    plt.close()


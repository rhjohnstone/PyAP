import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
import pyap_setup as ps
import sys


randn = npr.randn
sqrt = np.sqrt
def sample_from_N_IG(eta):
    mu, nu, alpha, beta = eta
    sigma_squared_sample = invgamma.rvs(alpha, scale=beta)
    m_sample = norm.rvs(mu, scale=sqrt(sigma_squared_sample/nu))
    return m_sample, sigma_squared_sample


fs = 16
phi = 1.61803398875
fig_y = 5

model = 2

if model==1:
    label="hodgkin_huxley"
elif model==2:
    label = "beeler_reuter"
elif model==3:
    label = "luo_rudy"
elif model==4:
    label = "ten_tusscher"
elif model==5:
    label = "ohara"
elif model==6:
    label = "davies"
elif model==7:
    label = "paci"
    
    
original_gs, g_parameters, model_name = ps.get_original_params(model)
num_gs = len(original_gs)

T = 10000

m_true = np.log(original_gs)
sigma2_true = 0.04

mu = m_true
alpha = 4.*np.ones(num_gs)
beta = (alpha+1.) * sigma2_true
nu = 4.*beta / ((alpha+1.) * np.log(10)**2)

eta = np.vstack((mu, nu, alpha, beta)).T

print eta

num_pts = 501

for i in xrange(num_gs):
    print "{} / {}".format(i+1, num_gs)

    fig = plt.figure(figsize=(phi*fig_y,fig_y))

    """ax1 = fig.add_subplot(121)
    ax1.grid()
    offset = 3
    x = np.linspace(int(s)-offset, int(s)+offset, num_pts)
    ax1.plot(x, norm.pdf(x, loc=s, scale=1./np.sqrt(lamb)), lw=2)
    ax1.set_xlabel(r"$\mu$", fontsize=fs)
    ax1.set_ylabel("Prior pdf", fontsize=fs)
    ax1.axvline(mu_true[i], lw=2, color='red', label='true')
    ax1.legend()"""

    ax2 = fig.add_subplot(111)
    ax2.grid()
    x = np.linspace(1e-5, 0.5, num_pts)
    ax2.plot(x, invgamma.pdf(x, eta[i,2], scale=eta[i,3]), lw=2)
    ax2.set_xlabel(r"$\sigma^2$", fontsize=fs)
    ax2.set_ylabel("Prior pdf", fontsize=fs)
    ax2.axvline(sigma2_true, lw=2, color='red', label='true')
    ax2.legend()

    xmin = m_true[i] - 2*np.log(10)
    xmax = m_true[i] + 2*np.log(10)
    x = np.linspace(xmin, xmax, num_pts)
    y = np.zeros(num_pts)
    for T in xrange(T):
        mu_sample, sigma2_sample = sample_from_N_IG(eta[i, :])
        y += norm.pdf(x, loc=mu_sample, scale=np.sqrt(sigma2_sample))
    y /= T
    
    fig3 = plt.figure(figsize=(phi*fig_y,fig_y))
    ax3 = fig3.add_subplot(111)
    ax3.grid()
    ax3.plot(x, y, lw=2)
    ax3.axvline(m_true[i], lw=2, color='red', label='model')
    ax3.axvline(m_true[i]+np.log(10), color='red', ls='--', lw=2, label=r"model $\pm$ log(10)")
    ax3.axvline(m_true[i]-np.log(10), color='red', ls='--', lw=2)
    ax3.set_xlabel(r"log(${}$)".format(g_parameters[i]), fontsize=fs)
    ax3.set_ylabel("Prior pred. pdf", fontsize=fs-2)
    ax3.legend(loc=2, fontsize=fs-2)
    fig3.tight_layout()
    #fig3.savefig("/home/ross/Documents/dphil/chapters/inferring-conductances/{0}_prior_preds/{0}_prior_predictive_{1}.png".format(label, g_parameters[i]))
    plt.show()


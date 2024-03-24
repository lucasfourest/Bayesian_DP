import numpy as np
import torch

from scipy.stats import t, binom
from scipy.special import logsumexp




def get_epsilon(powers, privacy_costs, target_delta=1e-5):
    eps= np.min((privacy_costs - np.log(target_delta)) / powers)
    return eps, target_delta

def get_delta(powers, privacy_costs, target_eps=1):
    delta=np.min(np.exp(privacy_costs - powers * target_eps))
    return target_eps, delta

def get_cost(power,ldistr,rdistr,total_steps,q):
    
    holder_correction = total_steps
    c_L = log_binom_expect(power + 1, q, ldistr, rdistr)
    c_R = log_binom_expect(power + 1, q, rdistr, ldistr)
    logmgf_samples = torch.max(c_L, c_R).cpu().numpy()
    n_samples = np.size(logmgf_samples)

    max_logmgf = np.max(logmgf_samples)
    log_mgf_mean = -np.log(n_samples) + logsumexp(logmgf_samples)
    log_mgf_std = 0.5 * (2 * max_logmgf - np.log(n_samples) +\
                             np.log(np.sum(np.exp(2 * logmgf_samples - 2 * max_logmgf) -\
                                        np.exp(2 * log_mgf_mean - 2 * max_logmgf))))
    log_conf_pentalty = np.log(t.ppf(q=1 - 1e-16, df=n_samples-1)) + log_mgf_std - 0.5 * np.log(n_samples - 1)
    bdp = logsumexp([log_mgf_mean, log_conf_pentalty]) / holder_correction
    
    return bdp
    


def log_binom_expect(n, p, ldistr, rdistr):
        k = torch.arange(n + 1, dtype= float)
        log_binom_coefs = torch.tensor(binom.logpmf(k, n=n, p=p))
        return torch.logsumexp(log_binom_coefs + scaled_renyi_gaussian(k, ldistr, rdistr), dim=1)


def scaled_renyi_gaussian(alphas, ldistr, rdistr):
    lmu, lsigma = ldistr
    rmu, rsigma = rdistr
    
    distances = lmu - rmu
    distances = torch.norm(distances, p=2, dim=-1).view(-1).to(alphas)
    return torch.ger(distances**2, alphas * (alphas - 1) / (2 * lsigma**2))
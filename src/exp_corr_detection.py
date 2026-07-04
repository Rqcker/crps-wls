"""Detection power of s_hat under equicorrelated sensor-residual noise.

One might expect residual correlation (the eight models share the input
spectrum) to lower detection. It does the opposite: s_hat = c_s . m is a
contrast with c_s . 1 = 0, so a common (correlated) noise component cancels in
s_hat and detection sharpens; only theta_hat (c_theta . 1 = 1) keeps that
component and so loses accuracy.

Var(s_hat) = c_s^T Sigma c_s,  Sigma_ij = rho sigma_i sigma_j (i!=j), Sigma_ii=sigma_i^2.
"""
import numpy as np
from scipy.stats import norm

from src.agents import AGENTS
from src.source import wls_coefficients

THETA_MED = 19.9
FAR = 0.05

sig = np.array([a.sigma for a in AGENTS])
_, c_s = wls_coefficients(AGENTS)
z = norm.ppf(1.0 - FAR)

print("rho   sd(s_hat)   detect@20%   detect@50%   detect@100%")
for rho in [0.0, 0.3, 0.6, 0.9]:
    C = rho * np.ones((8, 8)) + (1.0 - rho) * np.eye(8)
    Sigma = np.outer(sig, sig) * C
    sd = float(np.sqrt(c_s @ Sigma @ c_s))
    dets = [float(1.0 - norm.cdf(z - (f * THETA_MED) / sd)) for f in (0.2, 0.5, 1.0)]
    print(f"{rho:.1f}   {sd:7.2f}   " + "   ".join(f"{d*100:5.1f}%" for d in dets))

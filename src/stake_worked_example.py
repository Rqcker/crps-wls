"""Closed-form checks behind two remarks in the paper.

A. Correlation sharpens detection:
   Equicorrelated  Sigma_rho = (1-rho) D + rho sigma sigma^T,  D = diag(sigma_i^2).
   Var(s_hat; rho) = c_s^T Sigma_rho c_s = (1-rho) v_s0 + rho (c_s . sigma)^2,
   linear in rho with slope (c_s . sigma)^2 - v_s0 < 0  iff  (c_s.sigma)^2 < v_s0.
   Since c_s . 1 = 0 and the sigma_i are nearly common, c_s . sigma is small, so
   the slope is negative: detection sharpens. Reversal example: noise correlated
   along the contrast direction c_s itself inflates Var(s_hat).

B. Worked stake example under the precision-weighted-mean (PWM) credit deployment:
   kappa = (w . alpha)/(w . 1) is the credited fraction of a manipulation s, so
   G(frac) = p_conv * kappa * frac * theta, M* = G / (P_detect - a)  (the stake
   equation in the paper).
"""
import numpy as np
from scipy.stats import norm

from src.agents import AGENTS
from src.source import wls_coefficients

FAR = 0.05
THETA_MED = 19.9                      # LUCAS mineral median (g/kg)
SOC_TO_CO2 = 3.9 * 44.0 / 12.0        # 14.3 tCO2/ha per g/kg
PRICE, AREA = 15.0, 1.0               # $/tCO2, ha
CONV = SOC_TO_CO2 * PRICE * AREA      # $ per g/kg credited

sig = np.array([a.sigma for a in AGENTS])
alp = np.array([a.alpha for a in AGENTS])
c_theta, c_s = wls_coefficients(AGENTS)
D = np.diag(sig**2)
z = norm.ppf(1.0 - FAR)

# ---- A. equicorrelation closed form ---------------------------------------
v_s0 = float(c_s @ D @ c_s)
cs_sig = float(c_s @ sig)
print("== A. Var(s_hat; rho) = (1-rho)*v_s0 + rho*(c_s.sigma)^2 ==")
print(f"v_s0 = c_s'Dc_s = {v_s0:.2f}   (sd {np.sqrt(v_s0):.2f})")
print(f"(c_s.sigma)^2  = {cs_sig**2:.2f}   (c_s.sigma = {cs_sig:+.3f})")
print(f"slope = {cs_sig**2 - v_s0:.2f}  -> negative, so Var falls with rho")
print("rho   closed-form Var   direct c_s'Sigma c_s   sd     detect@50%")
for rho in (0.0, 0.3, 0.6, 0.9):
    Sigma = np.outer(sig, sig) * (rho * np.ones((8, 8)) + (1 - rho) * np.eye(8))
    var_direct = float(c_s @ Sigma @ c_s)
    var_closed = (1 - rho) * v_s0 + rho * cs_sig**2
    sd = np.sqrt(var_direct)
    det = 1.0 - norm.cdf(z - 0.5 * THETA_MED / sd)
    print(f"{rho:.1f}   {var_closed:10.2f}   {var_direct:14.2f}   {sd:6.2f}   {det*100:5.1f}%")

# reversal check: rank-one noise along the contrast direction c_s
u = c_s / np.linalg.norm(c_s)
gamma = v_s0  # add noise of comparable size along u
var_rev = float(c_s @ (D + gamma * np.outer(u, u)) @ c_s)
print(f"reversal: Sigma = D + v_s0 * u u^T (u || c_s) -> Var(s_hat) = {var_rev:.2f} (vs {v_s0:.2f})")

# ---- B. PWM-credit worked example ------------------------------------------
w = 1.0 / sig**2
kappa = float(w @ alp / w.sum())
print("\n== B. worked example, credit on precision-weighted mean ==")
print(f"kappa = w.alpha/w.1 = {kappa:.3f}  (credited fraction of manipulation)")
for frac in (0.20, 0.50, 1.00):
    s = frac * THETA_MED
    p_det = float(1.0 - norm.cdf(z - s / np.sqrt(v_s0)))
    G = CONV * kappa * s
    M = G / (p_det - FAR)                 # Eq. (12); deterrence uses power above size
    honest = CONV * THETA_MED
    manip_net = honest + G - p_det * M    # manipulator: keeps gain, slashed at P_det
    honest_net = honest - FAR * M         # honest source: slashed at the false-alarm rate
    print(f"frac={frac:.2f}: s={s:5.2f}  P_det={p_det*100:5.1f}%  mult={1/(p_det-FAR):5.1f}x"
          f"  G=${G:7.2f}  M*=${M:9.2f}  honest=${honest:.2f}"
          f"  manip net=${manip_net:.2f}  honest net=${honest_net:.2f}")
print("(per-round dollars at the median parcel; theta-hat-credit deployment has G=0, no stake)")

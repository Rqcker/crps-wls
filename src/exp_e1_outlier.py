"""Outlier-sensor ablation: how much of the detection/deterrence rests on the
single near-useless CNN-LSTM-ECA sensor (alpha=0.004, sigma=17.98)?

Reuses the paper's own wls_covariance and manipulation_detection_power so the
numbers match Section 5. Compares the full eight-sensor ensemble against the
seven-sensor ensemble with CNN-LSTM-ECA removed.
"""
import numpy as np

from src.agents import AGENTS
from src.source import wls_covariance, manipulation_detection_power

THETA_MED = 19.9          # median LUCAS mineral OC (g/kg), as in Section 5
FRACS = [0.2, 0.5, 1.0]   # manipulation as a fraction of theta
FAR = 0.05


def summarize(agents, label):
    alphas = sorted(a.alpha for a in agents)
    cov = wls_covariance(agents)
    v_s = float(cov[1, 1])
    sd = float(np.sqrt(v_s))
    print(f"\n[{label}]  N={len(agents)}")
    print(f"  alpha span : {min(alphas):.3f} .. {max(alphas):.3f}")
    print(f"  Var(s_hat) : {v_s:.2f}   sd(s_hat)=sqrt(v_s) : {sd:.2f} g/kg "
          f"(theta_med={THETA_MED})")
    for f in FRACS:
        p = manipulation_detection_power(agents, f * THETA_MED, FAR)
        mult = f"{1.0 / p:.1f}x" if p > 1e-9 else "inf"
        print(f"  {int(f*100):>3}% manip (s={f*THETA_MED:5.1f}): "
              f"P_detect={p*100:5.1f}%   stake M*/G={mult}")
    return sd


sd_full = summarize(AGENTS, "full 8 sensors")
agents7 = [a for a in AGENTS if a.name != "CNN-LSTM-ECA"]
sd_drop = summarize(agents7, "7 sensors (drop CNN-LSTM-ECA, alpha=0.004)")

print(f"\nsd(s_hat) ratio (drop / full) = {sd_drop / sd_full:.3f}")
print("Interpretation: if dropping the outlier barely changes sd(s_hat) and "
      "detection, the diversity does NOT rest on that one sensor.")

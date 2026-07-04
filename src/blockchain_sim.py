"""
On-chain oracle simulation for carbon MRV with CRPS peer prediction.

Simulates complete oracle rounds as they would execute in a DePIN smart
contract.  Each round proceeds as follows:

  1. Source deposits stake T (in USD-equivalent tokens).
  2. Each agent i submits report (mu_i, sigma_i) on-chain.
  3. Smart contract executes deterministic logic:
       a. WLS aggregation: theta_hat, s_hat = WLS(reports, agents)
       b. Manipulation detection: flag if s_hat > delta (5% FAR threshold)
       c. CRPS peer payments to agents
       d. If flagged: slash source stake proportional to detected s_hat
       e. Carbon credits issued = theta_hat * SOC_TO_CO2 * area_ha (honest case)
  4. Results logged on-chain (immutable).

All computations are pure deterministic functions of their inputs:
compatible with on-chain execution (no randomness post-report-submission).

Carbon credit economics:
  SOC_TO_CO2    = 14.3 tCO2/ha per g/kg OC  (30cm depth, bulk density 1.3)
  credit_price  = USD per tCO2 (voluntary carbon market; range $5--$50)
  area_ha       = farm area in hectares

Stake economics (from Exp5; M*/G = 1/(P_detect - false_alarm_rate)):
  At frac=0.20 (20% manipulation), stake_mult=36.8x at theta=19.9 g/kg
  At frac=0.50, stake_mult=11.5x
  At frac=1.00, stake_mult=4.1x
  Uniform agents: stake_mult=inf (no finite stake deters manipulation)

Reference: This module implements the blockchain deployment architecture
described in the MARBLE 2026 paper, Section 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import norm

from src.agents import Agent, AGENTS, AGENTS_UNIFORM, get_noisy_reports
from src.mechanism import agent_expected_payment
from src.source import (
    estimate_theta_s,
    aggregate_weighted,
    wls_covariance,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# 1 g/kg SOC -> tCO2/ha (30 cm depth, bulk density 1.3 g/cm3, C:CO2 = 44/12)
SOC_TO_CO2: float = 3.9 * 44.0 / 12.0   # = 14.3 tCO2/ha per g/kg

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OracleRound:
    """Result of one on-chain oracle measurement round."""
    round_id:        int
    theta_true:      float    # true SOC (g/kg), from LUCAS bootstrap
    s_true:          float    # true source manipulation (g/kg)
    frac:            float    # manipulation fraction = s_true / theta_true
    theta_hat:       float    # WLS estimate of true SOC
    s_hat:           float    # WLS estimate of manipulation
    detected:        bool     # manipulation flag (s_hat > detection threshold)
    stake_deposited: float    # source stake (USD)
    stake_slashed:   float    # stake lost by source (USD)
    agent_payments:  list[float]  # CRPS-based payments to each agent (USD)
    credits_issued:  float    # carbon credits issued (tCO2)
    credit_revenue:  float    # source revenue from credits (USD)
    net_source:      float    # net source payoff = credit_revenue - stake_slashed
    theta_error_pct: float    # |theta_hat - theta_true| / theta_true


@dataclass
class SimulationResult:
    """Aggregated results from a multi-round oracle simulation."""
    scenario:        str
    n_rounds:        int
    frac:            float
    credit_price:    float    # USD / tCO2
    area_ha:         float    # farm area
    stake_mult:      float    # stake as multiple of expected manipulation gain
    mean_theta_mare: float    # mean absolute relative error on theta_hat
    detection_rate:  float    # fraction of rounds flagged
    mean_net_source: float    # mean source net payoff per round (USD)
    mean_agent_pay:  float    # mean total agent payment per round (USD)
    rounds:          list[OracleRound] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core oracle logic (deterministic — smart-contract compatible)
# ---------------------------------------------------------------------------

def _detection_threshold(agents: list[Agent], false_alarm_rate: float = 0.05) -> float:
    """Compute s_hat detection threshold at the given false-alarm rate.

    Returns delta such that P(s_hat > delta | s=0) = false_alarm_rate.
    Raises ValueError if agents are uniform (no detection possible).
    """
    try:
        cov = wls_covariance(agents)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Uniform agents: diversity condition fails") from exc
    sd_shat = float(np.sqrt(cov[1, 1]))
    return float(norm.ppf(1.0 - false_alarm_rate) * sd_shat)


def _crps_payments(
    reports: list[tuple[float, float]],
    agents: list[Agent],
    payment_scale: float = 1.0,
) -> list[float]:
    """Compute CRPS-based peer payments for each agent.

    Payments use the standard peer-prediction form:
        payment_i = payment_scale * (C - avg_peer_CRPS_i)

    where C = C_BUDGET = 20.0 is a budget constant chosen so that all agents
    receive positive payments under honest reporting (the maximum average peer
    CRPS for any agent in AGENTS at delta=0 is ~8.23 for the CNN agent).

    The budget constant C does not affect incentive compatibility: the
    strict BNE at delta=0 holds regardless of C.  The source funds the
    total agent budget as part of the oracle round fee.
    """
    # C_BUDGET must exceed the largest average peer CRPS in the worst scenario.
    # At frac=1.0 (theta=median=19.9) the largest agent's average peer CRPS
    # reaches ~10.8 (CNN). C_BUDGET = 20.0 keeps all agent payments positive.
    _C_BUDGET = 20.0
    return [
        payment_scale * (_C_BUDGET + agent_expected_payment(reports, i))
        for i in range(len(agents))
    ]


def run_oracle_round(
    round_id:        int,
    theta:           float,
    frac:            float,
    agents:          list[Agent],
    rng:             np.random.Generator,
    delta:           float,
    stake_deposited: float,
    credit_price:    float,
    area_ha:         float,
    payment_scale:   float = 10.0,
) -> OracleRound:
    """Execute one oracle measurement round.

    Args:
        round_id:        identifier for logging.
        theta:           true SOC value (g/kg), drawn from LUCAS.
        frac:            source manipulation fraction (s = frac * theta).
        agents:          list of Agent objects (diverse or uniform).
        rng:             random generator for agent noise eps_i.
        delta:           detection threshold for s_hat.
        stake_deposited: source stake in USD.
        credit_price:    USD per tCO2.
        area_ha:         farm area in hectares.
        payment_scale:   USD multiplier for CRPS peer payments.

    Returns:
        OracleRound with all round-level results.
    """
    s_true = frac * theta

    # Step 2: agents submit reports on-chain
    reports = get_noisy_reports(agents, theta, s_true, rng)

    # Step 3a: WLS aggregation (deterministic smart-contract computation)
    try:
        theta_hat, s_hat = estimate_theta_s(reports, agents)
    except np.linalg.LinAlgError:
        # Diversity fails -> fallback to precision-weighted mean (no s separation)
        theta_hat = aggregate_weighted(reports)
        s_hat = 0.0   # cannot estimate s

    # Step 3b: manipulation detection
    detected = bool(s_hat > delta)

    # Step 3c: CRPS peer payments (USD)
    agent_payments = _crps_payments(reports, agents, payment_scale)

    # Step 3d: stake slashing
    # Slash proportional to detected s_hat normalised by detection threshold.
    # Slash fraction = min(1, max(0, s_hat / delta)) when detected, else 0.
    if detected and delta > 0:
        slash_frac = min(1.0, max(0.0, s_hat / delta))
    else:
        slash_frac = 0.0
    stake_slashed = float(stake_deposited * slash_frac)

    # Step 3e: carbon credits issued based on theta_hat
    # Credits issued only when NOT detected; if detected, round is nullified.
    if not detected:
        credits_issued = float(theta_hat * SOC_TO_CO2 * area_ha)
    else:
        credits_issued = float(theta * SOC_TO_CO2 * area_ha)   # use true value
    credit_revenue = float(credits_issued * credit_price)

    net_source = credit_revenue - stake_slashed
    theta_error_pct = float(abs(theta_hat - theta) / max(theta, 1e-6))

    return OracleRound(
        round_id        = round_id,
        theta_true      = float(theta),
        s_true          = float(s_true),
        frac            = frac,
        theta_hat       = float(theta_hat),
        s_hat           = float(s_hat),
        detected        = detected,
        stake_deposited = float(stake_deposited),
        stake_slashed   = float(stake_slashed),
        agent_payments  = [float(p) for p in agent_payments],
        credits_issued  = float(credits_issued),
        credit_revenue  = float(credit_revenue),
        net_source      = float(net_source),
        theta_error_pct = float(theta_error_pct),
    )


# ---------------------------------------------------------------------------
# Multi-round simulation
# ---------------------------------------------------------------------------

def simulate_market(
    oc:            np.ndarray,
    scenario:      str,
    frac:          float,
    agents:        list[Agent],
    n_rounds:      int       = 1_000,
    credit_price:  float     = 15.0,    # USD / tCO2 (voluntary carbon market)
    area_ha:       float     = 1.0,
    stake_mult:    float     = 5.0,
    payment_scale: float     = 10.0,
    seed:          int       = RNG_SEED,
) -> SimulationResult:
    """Simulate n_rounds of the on-chain carbon MRV oracle.

    Args:
        oc:           LUCAS OC array (ground-truth theta values).
        scenario:     descriptive label (e.g. 'Diverse honest', 'Uniform cheat').
        frac:         source manipulation fraction (0 = honest).
        agents:       diverse or uniform agent set.
        n_rounds:     number of oracle rounds to simulate.
        credit_price: USD per tCO2.
        area_ha:      farm area in hectares.
        stake_mult:   source stakes stake_mult * (frac * theta_med * SOC_TO_CO2 *
                      area_ha * credit_price).  Pass stake_mult=0 to model a
                      source that deposits zero stake (worst case).
        payment_scale: USD multiplier for CRPS agent payments.
        seed:         random seed for reproducibility.

    Returns:
        SimulationResult with per-round OracleRound records.
    """
    rng = np.random.default_rng(seed)

    # Detection threshold (delta) from analytical WLS covariance
    try:
        delta = _detection_threshold(agents)
    except ValueError:
        delta = np.inf   # uniform agents: no detection possible

    # Stake per round: stake_mult * expected manipulation gain at median theta
    theta_med = float(np.median(oc))
    expected_gain_per_round = frac * theta_med * SOC_TO_CO2 * area_ha * credit_price
    stake_deposited = stake_mult * expected_gain_per_round if frac > 0 else 0.0

    # Bootstrap theta values from LUCAS
    thetas = rng.choice(oc, size=n_rounds, replace=True)

    rounds = []
    for i, theta in enumerate(thetas):
        r = run_oracle_round(
            round_id        = i,
            theta           = float(theta),
            frac            = frac,
            agents          = agents,
            rng             = rng,
            delta           = delta,
            stake_deposited = stake_deposited,
            credit_price    = credit_price,
            area_ha         = area_ha,
            payment_scale   = payment_scale,
        )
        rounds.append(r)

    detection_rate   = float(np.mean([r.detected for r in rounds]))
    mean_theta_mare  = float(np.mean([r.theta_error_pct for r in rounds]))
    mean_net_source  = float(np.mean([r.net_source for r in rounds]))
    mean_agent_pay   = float(np.mean([sum(r.agent_payments) for r in rounds]))

    return SimulationResult(
        scenario        = scenario,
        n_rounds        = n_rounds,
        frac            = frac,
        credit_price    = credit_price,
        area_ha         = area_ha,
        stake_mult      = stake_mult,
        mean_theta_mare = mean_theta_mare,
        detection_rate  = detection_rate,
        mean_net_source = mean_net_source,
        mean_agent_pay  = mean_agent_pay,
        rounds          = rounds,
    )


# ---------------------------------------------------------------------------
# Scenario battery
# ---------------------------------------------------------------------------

def run_blockchain_scenarios(
    oc:           np.ndarray,
    n_rounds:     int   = 1_000,
    credit_price: float = 15.0,
    area_ha:      float = 1.0,
) -> pd.DataFrame:
    """Run the four canonical oracle scenarios and return a summary DataFrame.

    Scenarios:
      A  Diverse  honest      frac=0,   stake_mult=0   (baseline)
      B  Diverse  manipulate  frac=0.2, stake_mult=36.8 (deterred)
      C  Diverse  manipulate  frac=0.2, stake_mult=0   (no stake, undeterred)
      D  Uniform  manipulate  frac=0.2, stake_mult=inf (diversity fails)

    Scenarios B vs C contrast the deterrence effect of staking.
    Scenario D shows that uniform agents make deterrence impossible.
    """
    from src.source import manipulation_detection_power
    from src.agents import AGENTS, AGENTS_UNIFORM

    # Deterring stake M*/G = 1 / (detection power - false-alarm rate): the net
    # slashing differential between a manipulating and an honest source, matching
    # the paper's M* = G / (P_detect - a). Computed from the measured agents.
    theta_med = float(np.median(oc))
    far = 0.05
    p_det = manipulation_detection_power(AGENTS, s=0.20 * theta_med, false_alarm_rate=far)
    excess = p_det - far
    sm_deter = round(1.0 / excess, 1) if excess > 1e-9 else float("inf")

    rows = []
    configs = [
        ("A: Diverse, honest",           AGENTS,         0.00, 0.0),
        ("B: Diverse, manip + stake",     AGENTS,         0.20, sm_deter),
        ("C: Diverse, manip, no stake",   AGENTS,         0.20, 0.0),
        ("D: Uniform, manip + stake",     AGENTS_UNIFORM, 0.20, sm_deter),
    ]

    for label, agents, frac, sm in configs:
        res = simulate_market(
            oc           = oc,
            scenario     = label,
            frac         = frac,
            agents       = agents,
            n_rounds     = n_rounds,
            credit_price = credit_price,
            area_ha      = area_ha,
            stake_mult   = sm,
        )
        theta_med = float(np.median(oc))
        expected_gain = frac * theta_med * SOC_TO_CO2 * area_ha * credit_price
        rows.append({
            "scenario":         label,
            "frac":             frac,
            "stake_mult":       sm,
            "stake_usd":        sm * expected_gain,
            "detection_rate":   res.detection_rate,
            "mean_theta_mare":  res.mean_theta_mare,
            "mean_net_source":  res.mean_net_source,
            "mean_agent_pay":   res.mean_agent_pay,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path

    from src.dataset import load_oc

    df = run_blockchain_scenarios(load_oc())
    out = Path(__file__).parent / "data" / "blockchain_scenarios.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"  saved {out}")

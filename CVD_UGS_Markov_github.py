"""
Markov CVD cost-effectiveness model

Python version of the Excel CVD urban green space Markov model by Nesterovich (2026). This code was created with help of Claude AI. The Master's thesis with a detailed technical report can be found here: https://lup.lub.lu.se/student-papers/record/9236862

A 6-state Markov cohort model (healthy **H** / acute MI (**AMI**) / acute stroke (**AS**) /
post-MI (**PMI**) / post-stroke (**PS**) / **D**ead) run for 16 age cohorts, by urban green space (UGS) perceived sensory dimensions (PSD) scenarios. It captures the impact of UGS improvements on CVD through physical activity. Societal costs, QALYs, and potential cost-effectiveness are the results of the current model.
"""

import bisect
from collections import namedtuple

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df_tp = pd.read_csv("Markov_tp_CVD.csv") # age represent midpoint of 5-year age bands
print(df_tp.head())

# ## 2. General setup

cycle_length = 1  # cycle length equal to one year (use 1/12 for monthly)

# Age at baseline, stratified (midpoint of the 5-year age band)
n_age_init = np.arange(22, 98, 5)

# Every starting age must exist in the CSV
assert np.all(np.isin(n_age_init, df_tp["age"].values)), \
    "some cohort starting ages are not in df_tp" # it is return if some ages are not in the csv file

n_age_max = 100  # maximum age of follow-up
n_cycles = ((n_age_max - n_age_init) / cycle_length).astype(int)  # time horizon per cohort

discount = 0.03 # this will be used for costs and QALYs according to Swedish standards

# The 6 health states of the model
v_names_states = ["H", "AMI", "AS", "PMI", "PS", "D"]
disease_states = [s for s in v_names_states if s != "D"]
STATE_IDX = {s: i for i, s in enumerate(v_names_states)} # this maps each health state to its numerical index
n_states = len(v_names_states)

# Starting distribution: whole cohort begins Healthy
v_s_init = np.array([1, 0, 0, 0, 0, 0], dtype=float)

# One record per cohort, so the rest of the notebook can just do "for c in COHORTS"
# instead of repeating `n_age_init[j]`, `int(n_cycles[j])`, etc. everywhere.
Cohort = namedtuple("Cohort", "name age0 n_cyc start_row0")
COHORTS = [
    Cohort(f"age_{age}", int(age), int(n_cyc), int(age - n_age_init[0]))
    for age, n_cyc in zip(n_age_init, n_cycles)
]
COHORT_NAMES = [c.name for c in COHORTS]

def make_wcc(n):
    """Half-cycle correction."""
    w = np.concat(([0.5], np.ones(n - 1), [0.5]))
    assert np.isclose(w.sum(), n)
    return w # for example, 78 cycles will look like: 0.5 + 77 * 1 + 0.5

# Within-cycle correction (WCC) weights, per cohort
v_wcc_list = {c.name: make_wcc(c.n_cyc) for c in COHORTS}

print(f"{len(COHORTS)} cohorts set up, ages {n_age_init.min()}-{n_age_init.max()}, "
      f"{n_cycles.min()}-{n_cycles.max()} cycles each.")

# ## 3. Transition arrays

def build_transition_array(start_row0, n_cyc, adj_MI=1.0, adj_stroke=1.0, adj_mort=1.0):
    """
    Build an (n_states, n_states, n_cyc) transition-probability array for one
    cohort, optionally re-scaling the Healthy -> {AMI, AS, D} probabilities by
    a PA-scenario adjustment factor (adj_MI / adj_stroke / adj_mort).

    start_row0 : 0-indexed row in df_tp corresponding to this cohort's
                 starting age.
    """
    a_P = np.zeros((n_states, n_states, n_cyc))

    for i in range(n_cyc):
        row_i = start_row0 + i  # 0-indexed row in df_tp for this cycle

        p_H_AMI = df_tp["p_H_AMI"].iloc[row_i] * adj_MI
        p_H_AS = df_tp["p_H_AS"].iloc[row_i] * adj_stroke
        p_H_D = df_tp["p_H_D"].iloc[row_i] * adj_mort
        p_AMI_D = df_tp["p_AMI_D"].iloc[row_i]
        p_AS_D = df_tp["p_AS_D"].iloc[row_i]
        p_PMI_D = df_tp["p_PMI_D"].iloc[row_i]
        p_PS_D = df_tp["p_PS_D"].iloc[row_i]

        # From Healthy
        a_P[STATE_IDX["H"], STATE_IDX["AMI"], i] = p_H_AMI
        a_P[STATE_IDX["H"], STATE_IDX["AS"], i] = p_H_AS
        a_P[STATE_IDX["H"], STATE_IDX["D"], i] = p_H_D
        a_P[STATE_IDX["H"], STATE_IDX["H"], i] = 1 - p_H_AMI - p_H_AS - p_H_D

        # From acute MI
        a_P[STATE_IDX["AMI"], STATE_IDX["PMI"], i] = 1 - p_AMI_D
        a_P[STATE_IDX["AMI"], STATE_IDX["D"], i] = p_AMI_D

        # From acute stroke
        a_P[STATE_IDX["AS"], STATE_IDX["PS"], i] = 1 - p_AS_D
        a_P[STATE_IDX["AS"], STATE_IDX["D"], i] = p_AS_D

        # From post-MI
        a_P[STATE_IDX["PMI"], STATE_IDX["PMI"], i] = 1 - p_PMI_D
        a_P[STATE_IDX["PMI"], STATE_IDX["D"], i] = p_PMI_D

        # From post-stroke
        a_P[STATE_IDX["PS"], STATE_IDX["PS"], i] = 1 - p_PS_D
        a_P[STATE_IDX["PS"], STATE_IDX["D"], i] = p_PS_D

        # Dead is absorbing
        a_P[STATE_IDX["D"], STATE_IDX["D"], i] = 1

    return a_P

def check_P(a_P, label):
    """Sanity-check a transition array: no NAs, values in [0, 1], rows sum to 1."""
    if np.isnan(a_P).any():
        raise ValueError(f"{label}: transition array contains NA")
    if np.any((a_P < -1e-10) | (a_P > 1 + 1e-10 )):
        raise ValueError(f"{label}: probability outside [0, 1]")
    row_sums = a_P.sum(axis=1)  # sum over "to" state, for every (from_state, cycle)
    # Using a small tolerance for floating-point comparison
    if np.any(np.abs(row_sums - 1) > 1e-8):
        raise ValueError(f"{label}: rows do not sum to 1")
    return True

l_a_P = {c.name: build_transition_array(c.start_row0, c.n_cyc) for c in COHORTS}

for c in COHORTS:
    check_P(l_a_P[c.name], c.name)
print(f"✓ All {len(COHORTS)} transition arrays passed the checks (valid probabilities, rows sum to 1).")

# ## 4. Markov traces

def build_markov_trace(a_P, n_cyc):
  global v_s_init
  """Propagate the cohort distribution through the transition array."""
  m_M = np.zeros((n_cyc + 1, n_states))
  m_M[0, :] = v_s_init
  for i in range(n_cyc):
      m_M[i + 1, :] = m_M[i, :] @ a_P[:, :, i]
  return m_M

l_m_M = {c.name: build_markov_trace(l_a_P[c.name], c.n_cyc) for c in COHORTS}

rows_ok = all(np.allclose(tm.sum(axis=1), 1, atol=0) for tm in l_m_M.values())
deaths_ok = np.all(np.diff(l_m_M["age_22"][:, STATE_IDX["D"]]) >= 0)
print(f"✓ Trace rows sum to 1: {rows_ok}")
print(f"✓ Deaths never decrease (age_22 cohort): {deaths_ok}")

#Trace check function
def check_trace(m_M, label=""):
  if np.any(m_M < -1e-12):
    raise ValueError(f"{label}: negative proportion in trace")
  if np.any(np.abs(m_M.sum(axis=1) - 1) > 1e-10):
    raise ValueError(f"{label}: trace rows do not sum to 1")
  if np.any(np.diff(m_M[:, STATE_IDX["D"]]) < -1e-12):
    raise ValueError(f"{label}: dead proportion decreases")
  return True

def trace_table(cohort_name, scenario=None, every=1):
    """
    Turn a Markov trace array into a readable age-by-state table
    (proportions shown as percentages).

    every : show every Nth cycle (e.g. every=5 for a 5-yearly snapshot)
            instead of every single year.
    """
    m_M = l_m_M_scen[scenario][cohort_name] if scenario else l_m_M[cohort_name]
    age0 = int(cohort_name.split("_")[1])
    ages = age0 + np.arange(m_M.shape[0])

    df = pd.DataFrame(m_M, index=ages, columns=v_names_states)
    df.index.name = "age"
    return df.iloc[::every]

# Full trace for the age-57 cohort, base case (one row per year of age)
print(trace_table("age_57").apply(lambda col: col.map("{:.1%}".format)))

# ## 5. PSD scenarios: OR for effect on physical activity (PA)

df_or_psd = pd.DataFrame({
    "scenario": ["0 PSDs", "1 PSD", "2 PSDs", "3 PSDs", "4-5 PSDs"],
    "OR": [1.00, 1.08, 1.13, 1.41, 1.44],
    "CI_low": [1.00, 1.02, 1.04, 1.26, 1.24],
    "CI_high": [1.00, 1.15, 1.24, 1.57, 1.66],
})
print(df_or_psd)

# HRs for PA effects on outcomes (with 95% CIs)

# All-cause mortality (Lear et al., 2017)
hr_mort_med = {"HR": 0.74, "CI_low": 0.66, "CI_high": 0.84}   # Medium vs. Low
hr_mort_high = {"HR": 0.58, "CI_low": 0.51, "CI_high": 0.65}  # High vs. Low

# MI incidence (Kazemi et al., 2024) - same for Medium and High vs. Low
hr_MI_med = {"HR": 0.83, "CI_low": 0.76, "CI_high": 0.91}
hr_MI_high = {"HR": 0.83, "CI_low": 0.76, "CI_high": 0.91}

# Stroke incidence (Kazemi et al., 2024) - same for Medium and High vs. Low
hr_stroke_med = {"HR": 0.85, "CI_low": 0.79, "CI_high": 0.91}
hr_stroke_high = {"HR": 0.85, "CI_low": 0.79, "CI_high": 0.91}

# Step 1: Baseline PA distributions by age group (85+ uses 65-84 values)
df_pa = pd.DataFrame({
    "age_group": ["16-29", "30-44", "45-64", "65-84", "85+"],
    "prop_low": [0.321, 0.334, 0.385, 0.424, 0.424],
    "prop_med": [0.310, 0.318, 0.296, 0.275, 0.275],
    "prop_high": [0.369, 0.348, 0.319, 0.301, 0.301],
})

# Step 2: Map starting ages to PA age groups
def age_to_pa_group(age):
    if age < 30:
        return "16-29"
    elif age < 45:
        return "30-44"
    elif age < 65:
        return "45-64"
    elif age < 85:
        return "65-84"
    else:
        return "85+"

# Step 3: Ordinal (proportional-odds) shift by OR
def apply_ordinal_or(pa_base, OR):
  def plogis(x):
    return 1.0 / (1.0 + np.exp(-x)) # transforming into a probability
  def qlogis(p):
    return np.log(p / (1.0 - p)) # transforming into odds
  p_cum_med = pa_base["prop_med"] + pa_base["prop_high"]    # P(PA >= Medium)
  p_cum_high = pa_base["prop_high"]                         # P(PA >= High)

  # Shift cumulative log-odds by log(OR)
  new_p_cum_med = plogis(qlogis(p_cum_med) + np.log(OR))
  new_p_cum_high = plogis(qlogis(p_cum_high) + np.log(OR))

  return {
      "prop_low": 1 - new_p_cum_med,
      "prop_med": new_p_cum_med - new_p_cum_high,
      "prop_high": new_p_cum_high,
  }

# Test: check for age 22 cohort, scenario OR = 1.41 (3 PSDs)
pa_base_22 = df_pa[df_pa["age_group"] == age_to_pa_group(22)].iloc[0]

print(apply_ordinal_or(pa_base_22, OR=1.41))

def pa_or_adjustment(pa_base, pa_new, hr_med, hr_high):
  def pa_weighted_hr(pa, hr_med, hr_high):
    """P(low)*1 + P(med)*HR_med + P(high)*HR_high -- the same weighting used
    for MI, stroke and mortality, just with a different pair of HRs each time."""
    return pa["prop_low"] + pa["prop_med"] * hr_med["HR"] + pa["prop_high"] * hr_high["HR"]
  """Ratio of PA-weighted HR after the OR shift vs. before it."""
  return pa_weighted_hr(pa_new, hr_med, hr_high) / pa_weighted_hr(pa_base, hr_med, hr_high)

# 1. Helper function for adjustments
def pa_or_adjustment(pa_base, pa_new, hr_med, hr_high):
  def pa_weighted_hr(pa, hr_med, hr_high):
    return (
        pa["prop_low"]
        + pa["prop_med"] * hr_med["HR"]
        + pa["prop_high"] * hr_high["HR"]
    )

  return pa_weighted_hr(pa_new, hr_med, hr_high) / pa_weighted_hr(
      pa_base, hr_med, hr_high
  )

# 2. Single Unified Loop: Builds adjustments, transition arrays, and traces
l_adj = {}
l_a_P_scen = {}
l_m_M_scen = {}

for _, scen_row in df_or_psd.iterrows():
  scen, OR_s = scen_row["scenario"], scen_row["OR"]

  l_adj[scen] = {}
  l_a_P_scen[scen] = {}
  l_m_M_scen[scen] = {}

  for c in COHORTS:
    pa_base = df_pa[df_pa["age_group"] == age_to_pa_group(c.age0)].iloc[0]
    pa_new = apply_ordinal_or(pa_base, OR_s)

    # A. Build adjustment dictionary and store it in l_adj
    adj_dict = {
        "adj_MI": pa_or_adjustment(pa_base, pa_new, hr_MI_med, hr_MI_high),
        "adj_stroke": pa_or_adjustment(
            pa_base, pa_new, hr_stroke_med, hr_stroke_high
        ),
        "adj_mort": pa_or_adjustment(
            pa_base, pa_new, hr_mort_med, hr_mort_high
        ),
    }
    l_adj[scen][c.name] = adj_dict

    # B. Build and check transition array using ** unpacking
    a_P = build_transition_array(c.start_row0, c.n_cyc, **adj_dict)
    check_P(a_P, label=f"{scen} {c.name}")
    l_a_P_scen[scen][c.name] = a_P

    # C. Build and check Markov trace
    m_M = build_markov_trace(a_P, c.n_cyc)
    check_trace(m_M, label=f"{scen} {c.name}")
    l_m_M_scen[scen][c.name] = m_M

print(
    f"✓ All {len(df_or_psd)} scenarios and cohorts built and passed validation"
    " checks."
)

# 4. Preview: adjustment factors for the age-22 cohort across scenarios
preview_df = (
    pd.DataFrame([l_adj[s]["age_22"] for s in df_or_psd["scenario"]])
    .assign(scenario=list(df_or_psd["scenario"]))
    .set_index("scenario")
)
print(preview_df)



print(trace_table("age_57", scenario="4-5 PSDs").apply(lambda col: col.map("{:.1%}".format)))

# ## 6. Cost

# 1. PARAMETERS & CPI / EX
er_EUR_SEK_2025 = 11.07

v_cpi_ratios = {
    "2025_2016": 1.320924059,
    "2025_2013": 1.330892186,
    "2025_2024": 1.006816813,
    "2025_2023": 1.035372802,
    "2025_2015": 1.333907771,
    "2025_2008": 1.390439440,
    "2025_2012": 1.330299173,
}

ex16 = er_EUR_SEK_2025 * v_cpi_ratios["2025_2016"]
ex13 = er_EUR_SEK_2025 * v_cpi_ratios["2025_2013"]
ex24 = er_EUR_SEK_2025 * v_cpi_ratios["2025_2024"]

# 2. DIRECT MEDICAL COSTS (DMC) - Age Stratified
brk_DMC = [0, 30, 50, 67, 76, 80, 90] # breakpoints for age

v_DMC_H_raw = np.array([12890, 14498, 20973, 38352, 53574, 66518, 75478])  # 19-29,30-49,50-66,67-75,76-79,80-89,90p
v_DMC_H = v_DMC_H_raw * v_cpi_ratios["2025_2024"]

stroke_EXC = 189000 * v_cpi_ratios["2025_2008"] # acute stroke excess cost
v_DMC_AS = v_DMC_H + stroke_EXC # acute stroke cost

DMC_AS_NOAGE_EUR = 20242 # taken from a different source to calculate the relationship between acute and post-acute stroke
DMC_PS_NOAGE_EUR = 5438
v_DMC_PS = v_DMC_AS * (DMC_PS_NOAGE_EUR / DMC_AS_NOAGE_EUR) # post-stroke cost

DMC_AMI_NOAGE_EUR = 18411 # to calculate relationship between acute MI and acute stroke
v_DMC_AMI = v_DMC_AS * (DMC_AMI_NOAGE_EUR / DMC_AS_NOAGE_EUR) # acute MI cost

DMC_PMI_NOAGE_EUR = 6215
v_DMC_PMI = v_DMC_AMI * (DMC_PMI_NOAGE_EUR / DMC_AMI_NOAGE_EUR) # post-MI cost

# 3. NON-DIRECT MEDICAL COSTS (DNMC) - Age >= 80 (single value, "80p")
v_DNMC_AS = 18870 * ex16
v_DNMC_PS = 27473 * ex16
v_DNMC_AMI = DMC_AMI_NOAGE_EUR * (18870 / DMC_AS_NOAGE_EUR) * ex13
v_DNMC_PMI = v_DNMC_AMI * (v_DNMC_PS / v_DNMC_AS)
v_DNMC_H = v_DNMC_PMI / 3.4

# 4. INDIRECT COSTS (IC) - Age < 65 (single value, "22_64")
v_IC_AS = 6093 * ex16
v_IC_PS = 5292 * ex16
v_IC_AMI = 3670 * ex13
v_IC_PMI = 3682 * ex13
v_IC_H = 1807 * ex16
v_IC_D = 527781

# 5. END-OF-LIFE COSTS (EoLC)
v_EoLC = np.array([
    42053 * ex24,  # "<80"
    49091 * ex24,  # "80p"
])
brk_EoLC = [0, 80]

def lookup_age(age, values, breaks):
    """Equivalent of R's findInterval(age, breaks) indexing into `values`."""
    idx = bisect.bisect_right(breaks, age)  # 1-indexed bin, R-style
    return values[idx - 1]

def get_EoLC(age):
    return lookup_age(age, v_EoLC, brk_EoLC)

# State to cost type dictionaries
DMC = {"H": v_DMC_H, "AMI": v_DMC_AMI, "AS": v_DMC_AS,
       "PMI": v_DMC_PMI, "PS": v_DMC_PS}

DNMC = {"H": v_DNMC_H, "AMI": v_DNMC_AMI, "AS": v_DNMC_AS,
        "PMI": v_DNMC_PMI, "PS": v_DNMC_PS}

IC = {"H": v_IC_H, "AMI": v_IC_AMI, "AS": v_IC_AS,
      "PMI": v_IC_PMI, "PS": v_IC_PS, "D": v_IC_D}

def compute_costs(m_M, n_cyc, age0, v_wcc):
    """Compute per-cohort DMC / DNMC / IC / EoLC totals (undiscounted & discounted)."""
    v_DMC = np.zeros(n_cyc)
    v_DNMC = np.zeros(n_cyc)
    v_IC = np.zeros(n_cyc)
    v_EoLC = np.zeros(n_cyc)

    for i in range(n_cyc):
        age = age0 + i
        wcc = v_wcc[i]
        occ = {s: m_M[i, STATE_IDX[s]] for s in disease_states}  # state occupancy this cycle

        v_DMC[i] = wcc * sum(occ[s] * lookup_age(age, DMC[s], brk_DMC) for s in disease_states)

        if age >= 80:
            v_DNMC[i] = wcc * sum(occ[s] * DNMC[s] for s in disease_states)

        if age < 65:
            v_IC[i] = wcc * (
                sum(occ[s] * IC[s] for s in disease_states)
                + m_M[i, STATE_IDX["D"]] * IC["D"]
            )

        newly_dead = m_M[i + 1, STATE_IDX["D"]] - m_M[i, STATE_IDX["D"]]
        v_EoLC[i] = wcc * newly_dead * get_EoLC(age)

    discount_weights = 1.0 / (1.0 + discount) ** np.arange(n_cyc)
    parts = {"DMC": v_DMC, "DNMC": v_DNMC, "IC": v_IC, "EoLC": v_EoLC}
    total = sum(parts.values())

    out = {"total_cost": total.sum(), "total_discounted_cost": (total * discount_weights).sum()}
    for name, arr in parts.items():
        out[name] = arr.sum()
        out[f"{name}_discounted"] = (arr * discount_weights).sum()
    return out

l_m_C_scen = {
    scen: {
        c.name: compute_costs(l_m_M_scen[scen][c.name], c.n_cyc, c.age0, v_wcc_list[c.name])
        for c in COHORTS
    }
    for scen in df_or_psd["scenario"]
}

print("✓ Costs computed for all scenarios and cohorts.")

# Population weights
v_pop_weights = {
    "age_22": 70, "age_27": 73, "age_32": 92, "age_37": 88,
    "age_42": 79, "age_47": 77, "age_52": 81, "age_57": 81,
    "age_62": 74, "age_67": 67, "age_72": 63, "age_77": 64,
    "age_82": 46, "age_87": 27, "age_92": 13, "age_97": 4,
}

# Aggregate weighted discounted costs by scenario
v_total_discounted_cost_scen = {
    scen: sum(l_m_C_scen[scen][a]["total_discounted_cost"] * v_pop_weights[a] for a in v_pop_weights)
    for scen in l_m_C_scen
}

# Incremental cost
v_incremental_cost = {
    "1 PSD vs 0 PSDs": v_total_discounted_cost_scen["1 PSD"] - v_total_discounted_cost_scen["0 PSDs"],
    "2 PSDs vs 1 PSD": v_total_discounted_cost_scen["2 PSDs"] - v_total_discounted_cost_scen["1 PSD"],
    "3 PSDs vs 2 PSDs": v_total_discounted_cost_scen["3 PSDs"] - v_total_discounted_cost_scen["2 PSDs"],
    "4-5 PSDs vs 3 PSDs": v_total_discounted_cost_scen["4-5 PSDs"] - v_total_discounted_cost_scen["3 PSDs"],
}

print("✓ Discounted costs aggregated across the population.")

# ## 7. QALYs
# Since cycle length is 1 year, the EQ-5D scores remain the same (EQ-5D * 1).

## QALY, healthy
# Bands: <35 (uses 30-34), 35-39, 40-44, ..., 90-94, 95+
brk_QALY_H = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

QALY_NOAGE_H = 0.886

v_QALY_H = np.array([
    0.902, 0.910, 0.906, 0.904, 0.895, 0.894, 0.899,
    0.909, 0.899, 0.883, 0.852, 0.814, 0.761, 0.696,
])  # 30-34,35-39,40-44,45-49,50-54,55-59,60-64,65-69,70-74,75-79,80-84,85-89,90-94,95-104

## QALYs acute MI
QALY_NOAGE_AMI = 0.73
v_QALY_AMI = QALY_NOAGE_AMI * (v_QALY_H / QALY_NOAGE_H)

## QALY post-MI
QALY_NOAGE_PMI = 0.83
v_QALY_PMI = QALY_NOAGE_PMI * (v_QALY_H / QALY_NOAGE_H)

## QALY acute stroke
# Bands: 22-64, 65-74, 75-84, 85+
brk_QALY_stroke = [0, 65, 75, 85]

QALY_NOAGE_AS = 0.61
v_QALY_AS = np.array([0.71, 0.69, 0.60, 0.48])  # 22-64, 65-74, 75-84, 85p

## QALY post-stroke
QALY_NOAGE_PS = 0.66
v_QALY_PS = QALY_NOAGE_PS * (v_QALY_AS / QALY_NOAGE_AS)

# State to QALY type dictionaries
QALY = {"H": v_QALY_H, "AMI": v_QALY_AMI, "AS": v_QALY_AS,
        "PMI": v_QALY_PMI, "PS": v_QALY_PS}
QALY_BREAKS = {"H": brk_QALY_H, "AMI": brk_QALY_H, "AS": brk_QALY_stroke,
               "PMI": brk_QALY_H, "PS": brk_QALY_stroke}

def compute_qalys(m_M, n_cyc, age0, v_wcc):
    v_QALY = np.zeros(n_cyc)
    for i in range(n_cyc):
        age = age0 + i
        v_QALY[i] = v_wcc[i] * sum(
            m_M[i, STATE_IDX[s]] * lookup_age(age, QALY[s], QALY_BREAKS[s])
            for s in disease_states
        )

    discount_weights = 1.0 / (1.0 + discount) ** np.arange(n_cyc)
    return {"total_QALY": v_QALY.sum(), "total_discounted_QALY": (v_QALY * discount_weights).sum()}

l_m_Q_scen = {
    scen: {
        c.name: compute_qalys(l_m_M_scen[scen][c.name], c.n_cyc, c.age0, v_wcc_list[c.name])
        for c in COHORTS
    }
    for scen in df_or_psd["scenario"]
}

## Total QALYs and incremental QALYs
v_total_discounted_QALY_scen = {
    scen: sum(l_m_Q_scen[scen][a]["total_discounted_QALY"] * v_pop_weights[a] for a in v_pop_weights)
    for scen in l_m_Q_scen
}

v_incremental_QALY = {
    "1 PSD vs 0 PSDs": v_total_discounted_QALY_scen["1 PSD"] - v_total_discounted_QALY_scen["0 PSDs"],
    "2 PSDs vs 1 PSD": v_total_discounted_QALY_scen["2 PSDs"] - v_total_discounted_QALY_scen["1 PSD"],
    "3 PSDs vs 2 PSDs": v_total_discounted_QALY_scen["3 PSDs"] - v_total_discounted_QALY_scen["2 PSDs"],
    "4-5 PSDs vs 3 PSDs": v_total_discounted_QALY_scen["4-5 PSDs"] - v_total_discounted_QALY_scen["3 PSDs"]
}

print("✓ QALYs aggregated across the population.")

# WTP threshold
WTP = 500000

# Maximum allowed intervention cost (Net Monetary Benefit approach)
# Max cost = WTP * Incremental QALY - Incremental Cost
v_max_intervention_cost = {
    label: WTP * v_incremental_QALY[label] - v_incremental_cost[label]
    for label in v_incremental_QALY
}

# ## 8. Checks

wcc_ok = all(np.isclose(v_wcc_list[c.name].sum(), c.n_cyc) for c in COHORTS)
print(f"✓ WCC weights sum to the number of cycles for every cohort: {wcc_ok}")

# ## 9. Results summary
#
# Everything above builds the model; these tables are the final, readable output.

# --- Table 1: population-weighted discounted cost & QALYs by scenario ---
scenario_summary = pd.DataFrame({
    "scenario": list(df_or_psd["scenario"]),
    "OR": list(df_or_psd["OR"]),
    "discounted_cost": [v_total_discounted_cost_scen[s] for s in df_or_psd["scenario"]],
    "discounted_QALY": [v_total_discounted_QALY_scen[s] for s in df_or_psd["scenario"]],
}).set_index("scenario")

print("Cost & QALYs by PSD scenario (population-weighted, discounted)")
print(scenario_summary.to_string(formatters={"OR": "{:.2f}".format, "discounted_cost": "{:,.0f}".format, "discounted_QALY": "{:,.1f}".format}))

# --- Table 2: incremental comparisons & maximum affordable intervention cost ---
comparison_summary = pd.DataFrame({
    "comparison": list(v_incremental_cost.keys()),
    "incremental_cost": list(v_incremental_cost.values()),
    "incremental_QALY": list(v_incremental_QALY.values()),
    "max_intervention_cost": list(v_max_intervention_cost.values()),
}).set_index("comparison")

print(f"Incremental cost/QALYs and maximum affordable intervention cost (WTP = {WTP:,.0f}/QALY)")
print(comparison_summary.to_string(formatters={
    "incremental_cost": "{:,.0f}".format,
    "incremental_QALY": "{:,.1f}".format,
    "max_intervention_cost": "{:,.0f}".format,
}))

#Dicsounted total cost by type of cost
keys = {"DMC": "DMC_discounted", "DNMC": "DNMC_discounted", "IC": "IC_discounted",
        "EoLC": "EoLC_discounted"}

df_cost_by_type = pd.DataFrame({
    scen: {name: sum(l_m_C_scen[scen][a][key] * v_pop_weights[a] for a in v_pop_weights)
           for name, key in keys.items()}
    for scen in l_m_C_scen
}).T

print("\nDiscounted cost by type (population-weighted)")
print(df_cost_by_type.round(0).astype(int))

# Markov CVD cost-effectiveness model
# 6 states: H / AMI / AS / PMI / PS / D, 16 age cohorts, PSD scenario analysis,
# costs, QALYs and the maximum affordable intervention cost.

# 1. DATA & GENERAL SETUP --------------------------------------------------------
df_tp <- read.csv("C:/Users/37256/OneDrive/Desktop/Health economics and CEIE/Thesis/Markov_tp_CVD.csv")
head(df_tp)

cycle_length <- 1                        # one year
n_age_max    <- 100                      # maximum age
discount     <- 0.03                     # costs and QALYs (Swedish standard)

n_age_init <- seq(22, 97, by = 5)        # baseline age of each cohort
names(n_age_init) <- paste0("age_", n_age_init)
cohort_names <- names(n_age_init)
stopifnot("some cohort starting ages are not in df_tp" = all(n_age_init %in% df_tp$age))
n_cycles <- (n_age_max - n_age_init) / cycle_length   # keeps the cohort names

v_names_states <- c("H", "AMI", "AS", "PMI", "PS", "D")
disease_states <- setdiff(v_names_states, "D")
n_states <- length(v_names_states)
v_s_init <- c(1, 0, 0, 0, 0, 0)          # everyone starts Healthy

# Half-cycle correction weights 
make_wcc <- function(n) {
  w <- c(0.5, rep(1, n - 1), 0.5)
  stopifnot(isTRUE(all.equal(sum(w), n)))
  w
}
v_wcc_list <- lapply(n_cycles, make_wcc)

# Each cohort's results are kept in a list named by cohort; scenario results are
# a list (scenario) of lists (cohort). Map() runs a function over all cohorts.
by_cohort <- function(f, ...) Map(f, n_age_init, n_cycles, ...)

cat(length(n_age_init), "cohorts set up, ages", min(n_age_init), "-", max(n_age_init), "\n")

# 2. TRANSITION ARRAYS -----------------------------------------------------------
# Where each post-event state goes if the person does not die
next_state <- c(AMI = "PMI", AS = "PS", PMI = "PMI", PS = "PS")

build_transition_array <- function(age0, n_cyc, adj_MI = 1, adj_stroke = 1, adj_mort = 1) {
  tp <- df_tp[match(age0, df_tp$age) + seq_len(n_cyc) - 1, ]
  p_H <- cbind(AMI = tp$p_H_AMI * adj_MI,       # Healthy -> AMI / AS / D (scenario-adjusted)
               AS  = tp$p_H_AS  * adj_stroke,
               D   = tp$p_H_D   * adj_mort)

  a_P <- array(0, c(n_states, n_states, n_cyc),
               dimnames = list(from = v_names_states, to = v_names_states, cycle = NULL))
  a_P["H", colnames(p_H), ] <- t(p_H)
  a_P["H", "H", ] <- 1 - rowSums(p_H)

  for (s in names(next_state)) {                # AMI, AS, PMI, PS
    p_D <- tp[[paste0("p_", s, "_D")]]
    a_P[s, next_state[[s]], ] <- 1 - p_D
    a_P[s, "D", ] <- p_D
  }
  a_P["D", "D", ] <- 1                          # Dead is absorbing
  a_P
}

check_P <- function(a_P, label) {
  if (anyNA(a_P)) stop(label, ": transition array contains NA")
  if (any(a_P < -1e-10 | a_P > 1 + 1e-10)) stop(label, ": probability outside [0, 1]")
  if (any(abs(apply(a_P, c(1, 3), sum) - 1) > 1e-8)) stop(label, ": rows do not sum to 1")
  invisible(TRUE)
}

l_a_P <- by_cohort(build_transition_array)
invisible(Map(check_P, l_a_P, names(l_a_P)))
cat("All", length(l_a_P), "transition arrays passed the checks\n")

# 3. MARKOV TRACES ---------------------------------------------------------------
build_markov_trace <- function(a_P, n_cyc) {
  m_M <- matrix(0, n_cyc + 1, n_states, dimnames = list(NULL, v_names_states))
  m_M[1, ] <- v_s_init
  for (i in seq_len(n_cyc)) m_M[i + 1, ] <- m_M[i, ] %*% a_P[, , i]
  m_M
}

check_trace <- function(m_M, label = "") {
  if (any(m_M < -1e-12)) stop(label, ": negative proportion in trace")
  if (any(abs(rowSums(m_M) - 1) > 1e-10)) stop(label, ": trace rows do not sum to 1")
  if (any(diff(m_M[, "D"]) < -1e-12)) stop(label, ": dead proportion decreases")
  invisible(TRUE)
}

l_m_M <- Map(build_markov_trace, l_a_P, n_cycles)
invisible(Map(check_trace, l_m_M, names(l_m_M)))
cat("Base-case traces passed the checks\n")

# Age-by-state table in %, every `every`-th cycle
trace_table <- function(cohort, scenario = NULL, every = 1) {
  m_M <- if (is.null(scenario)) l_m_M[[cohort]] else l_m_M_scen[[scenario]][[cohort]]
  tab <- 100 * m_M
  rownames(tab) <- n_age_init[[cohort]] + seq_len(nrow(m_M)) - 1
  round(tab[seq(1, nrow(tab), by = every), , drop = FALSE], 1)
}
trace_table("age_57")

# 4. PSD SCENARIOS ---------------------------------------------------------------
df_or_psd <- data.frame(
  scenario = c("0 PSDs", "1 PSD", "2 PSDs", "3 PSDs", "4-5 PSDs"),
  OR       = c(1.00, 1.08, 1.13, 1.41, 1.44),
  CI_low   = c(1.00, 1.02, 1.04, 1.26, 1.24),
  CI_high  = c(1.00, 1.15, 1.24, 1.57, 1.66)
)
scenarios <- df_or_psd$scenario

# HRs for PA effects (Medium / High vs Low): mortality (Lear 2017), MI & stroke (Kazemi 2024)
df_hr <- data.frame(
  outcome = rep(c("mort", "MI", "stroke"), each = 2),
  level   = rep(c("med", "high"), times = 3),
  HR      = c(0.74, 0.58, 0.83, 0.83, 0.85, 0.85),
  CI_low  = c(0.66, 0.51, 0.76, 0.76, 0.79, 0.79),
  CI_high = c(0.84, 0.65, 0.91, 0.91, 0.91, 0.91)
)
l_hr <- split(setNames(df_hr$HR, df_hr$level), df_hr$outcome)  # $MI, $mort, $stroke: c(med, high)

# Baseline PA distribution by age group (85+ uses the 65-84 values)
m_pa <- rbind(
  "16-29" = c(low = 0.321, med = 0.310, high = 0.369),
  "30-44" = c(0.334, 0.318, 0.348),
  "45-64" = c(0.385, 0.296, 0.319),
  "65-84" = c(0.424, 0.275, 0.301),
  "85+"   = c(0.424, 0.275, 0.301)
)
age_to_pa_group <- function(age) {
  as.character(cut(age, c(-Inf, 30, 45, 65, 85, Inf), labels = rownames(m_pa), right = FALSE))
}

# Proportional-odds shift of the PA distribution by an odds ratio
apply_ordinal_or <- function(pa, OR) {
  p_cum <- plogis(qlogis(c(med = pa[["med"]] + pa[["high"]], high = pa[["high"]])) + log(OR))
  c(low = 1 - p_cum[["med"]], med = p_cum[["med"]] - p_cum[["high"]], high = p_cum[["high"]])
}
apply_ordinal_or(m_pa[age_to_pa_group(22), ], OR = 1.41)  # test: age 22, 3 PSDs

# Ratio of the PA-weighted HR after vs before the OR shift
pa_weighted_hr <- function(pa, hr) sum(pa * c(low = 1, hr[c("med", "high")]))
pa_or_adjustment <- function(pa_base, pa_new, hr) {
  pa_weighted_hr(pa_new, hr) / pa_weighted_hr(pa_base, hr)
}

# One scenario: adjustment factors, transition array and trace for every cohort
build_scenario <- function(OR_s) {
  by_cohort(function(age0, n_cyc) {
    pa_base <- m_pa[age_to_pa_group(age0), ]
    pa_new  <- apply_ordinal_or(pa_base, OR_s)
    adj <- sapply(l_hr, pa_or_adjustment, pa_base = pa_base, pa_new = pa_new)
    names(adj) <- paste0("adj_", names(adj))                      # adj_MI, adj_mort, adj_stroke
    a_P <- do.call(build_transition_array, c(list(age0, n_cyc), as.list(adj)))
    check_P(a_P, paste(OR_s, age0))
    m_M <- build_markov_trace(a_P, n_cyc)
    check_trace(m_M, paste(OR_s, age0))
    list(adj = adj, a_P = a_P, m_M = m_M)
  })
}
l_scen <- lapply(setNames(df_or_psd$OR, scenarios), build_scenario)

pluck <- function(l, el) lapply(l, lapply, `[[`, el)     # scenario -> cohort -> element
l_adj       <- pluck(l_scen, "adj")
l_m_M_scen  <- pluck(l_scen, "m_M")
cat("All", length(scenarios), "scenarios and cohorts built and passed validation\n")

t(sapply(l_adj, `[[`, "age_22"))                          # adjustment factors, age-22 cohort
trace_table("age_57", scenario = "4-5 PSDs")

# 5. COSTS -----------------------------------------------------------------------
er_EUR_SEK_2025 <- 11.07
cpi <- c("2025_2016" = 1.320924059, "2025_2013" = 1.330892186, "2025_2024" = 1.006816813,
         "2025_2023" = 1.035372802, "2025_2015" = 1.333907771, "2025_2008" = 1.390439440,
         "2025_2012" = 1.330299173)
fx16 <- er_EUR_SEK_2025 * cpi[["2025_2016"]]
fx13 <- er_EUR_SEK_2025 * cpi[["2025_2013"]]
fx24 <- er_EUR_SEK_2025 * cpi[["2025_2024"]]

# Direct medical costs (age-banded): 19-29, 30-49, 50-66, 67-75, 76-79, 80-89, 90+
brk_DMC <- c(0, 30, 50, 67, 76, 80, 90)
v_DMC_H  <- c(12890, 14498, 20973, 38352, 53574, 66518, 75478) * cpi[["2025_2024"]]
v_DMC_AS <- v_DMC_H + 189000 * cpi[["2025_2008"]]          # acute stroke = healthy + excess cost

# Cost ratios from a different (non-age-specific) source
DMC_AS_NOAGE_EUR  <- 20242
DMC_PS_NOAGE_EUR  <- 5438
DMC_AMI_NOAGE_EUR <- 18411
DMC_PMI_NOAGE_EUR <- 6215
v_DMC_PS  <- v_DMC_AS  * DMC_PS_NOAGE_EUR  / DMC_AS_NOAGE_EUR
v_DMC_AMI <- v_DMC_AS  * DMC_AMI_NOAGE_EUR / DMC_AS_NOAGE_EUR
v_DMC_PMI <- v_DMC_AMI * DMC_PMI_NOAGE_EUR / DMC_AMI_NOAGE_EUR

# Non-direct medical costs: flat, only applied from age 80
v_DNMC_AS  <- 18870 * fx16
v_DNMC_PS  <- 27473 * fx16
v_DNMC_AMI <- DMC_AMI_NOAGE_EUR * (18870 / DMC_AS_NOAGE_EUR) * fx13
v_DNMC_PMI <- v_DNMC_AMI * v_DNMC_PS / v_DNMC_AS
v_DNMC_H   <- v_DNMC_PMI / 3.4

# Indirect costs: flat, only applied below age 65 ("D" = dead people too)
v_IC <- c(H = 1807 * fx16, AMI = 3670 * fx13, AS = 6093 * fx16,
          PMI = 3682 * fx13, PS = 5292 * fx16, D = 527781)

# End-of-life costs: < 80 and 80+
v_EoLC   <- c(42053, 49091) * fx24
brk_EoLC <- c(0, 80)

DMC  <- cbind(H = v_DMC_H, AMI = v_DMC_AMI, AS = v_DMC_AS,
              PMI = v_DMC_PMI, PS = v_DMC_PS)[, disease_states]       # bands x states
DNMC <- c(H = v_DNMC_H, AMI = v_DNMC_AMI, AS = v_DNMC_AS,
          PMI = v_DNMC_PMI, PS = v_DNMC_PS)[disease_states]

# Undiscounted and discounted column totals of an (n_cyc x k) matrix of per-cycle values
discount_totals <- function(x) {
  dw <- (1 + discount)^-(seq_len(nrow(x)) - 1)
  rbind(undiscounted = colSums(x), discounted = colSums(x * dw))
}

# Returns a 2 x 5 matrix: {undiscounted, discounted} x {DMC, DNMC, IC, EoLC, total}
compute_costs <- function(age0, n_cyc, m_M, v_wcc) {
  cyc  <- seq_len(n_cyc)
  ages <- age0 + cyc - 1
  occ  <- m_M[cyc, disease_states]                         # state occupancy each cycle
  dead <- m_M[, "D"]

  parts <- cbind(
    DMC  = rowSums(occ * DMC[findInterval(ages, brk_DMC), ]),
    DNMC = (ages >= 80) * drop(occ %*% DNMC),
    IC   = (ages < 65) * (drop(occ %*% v_IC[disease_states]) + dead[cyc] * v_IC[["D"]]),
    EoLC = diff(dead) * v_EoLC[findInterval(ages, brk_EoLC)]   # newly dead x EoL cost
  ) * v_wcc[cyc]

  discount_totals(cbind(parts, total = rowSums(parts)))
}

l_m_C_scen <- lapply(l_m_M_scen, function(l_M) Map(compute_costs, n_age_init, n_cycles, l_M, v_wcc_list))
cat("Costs computed for all scenarios and cohorts\n")

# Population weights, and weighted sum of per-cohort results 
v_pop_weights <- c(age_22 = 70, age_27 = 73, age_32 = 92, age_37 = 88,
                   age_42 = 79, age_47 = 77, age_52 = 81, age_57 = 81,
                   age_62 = 74, age_67 = 67, age_72 = 63, age_77 = 64,
                   age_82 = 46, age_87 = 27, age_92 = 13, age_97 = 4)
pop_total <- function(l_res) Reduce(`+`, Map(`*`, l_res, v_pop_weights[names(l_res)]))

l_C_pop <- lapply(l_m_C_scen, pop_total)                   # per scenario: 2 x 5 matrix
v_total_discounted_cost_scen <- sapply(l_C_pop, `[`, "discounted", "total")

comparison <- paste(scenarios[-1], "vs", scenarios[-length(scenarios)])
v_incremental_cost <- setNames(diff(v_total_discounted_cost_scen[scenarios]), comparison)

# 6. QALYs -----------------------------------------------------------------------
# Healthy bands: <35 (uses 30-34), 35-39, ..., 90-94, 95+
brk_QALY_H <- c(0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
v_QALY_H <- c(0.902, 0.910, 0.906, 0.904, 0.895, 0.894, 0.899,
              0.909, 0.899, 0.883, 0.852, 0.814, 0.761, 0.696)
# Stroke bands: 22-64, 65-74, 75-84, 85+
brk_QALY_stroke <- c(0, 65, 75, 85)
v_QALY_AS <- c(0.71, 0.69, 0.60, 0.48)

q_noage   <- c(H = 0.886, AMI = 0.73, PMI = 0.83, AS = 0.61, PS = 0.66)
ref_state <- c(AMI = "H", PMI = "H", PS = "AS")            # state each QALY is scaled from

QALY        <- list(H = v_QALY_H, AS = v_QALY_AS)
QALY_BREAKS <- list(H = brk_QALY_H, AS = brk_QALY_stroke)
QALY[names(ref_state)] <- lapply(names(ref_state), function(s) {
  q_noage[[s]] / q_noage[[ref_state[[s]]]] * QALY[[ref_state[[s]]]]
})
QALY_BREAKS[names(ref_state)] <- QALY_BREAKS[ref_state]

# Returns a 2 x 1 matrix: {undiscounted, discounted} x QALY
compute_qalys <- function(age0, n_cyc, m_M, v_wcc) {
  cyc  <- seq_len(n_cyc)
  ages <- age0 + cyc - 1
  q <- vapply(disease_states, function(s) QALY[[s]][findInterval(ages, QALY_BREAKS[[s]])],
              numeric(n_cyc))                              # n_cyc x states
  discount_totals(cbind(QALY = v_wcc[cyc] * rowSums(m_M[cyc, disease_states] * q)))
}

l_m_Q_scen <- lapply(l_m_M_scen, function(l_M) Map(compute_qalys, n_age_init, n_cycles, l_M, v_wcc_list))
v_total_discounted_QALY_scen <- sapply(lapply(l_m_Q_scen, pop_total), `[`, "discounted", "QALY")
v_incremental_QALY <- setNames(diff(v_total_discounted_QALY_scen[scenarios]), comparison)
cat("QALYs aggregated across the population\n")

# 7. NET MONETARY BENEFIT (max intervention cost caluclation) --------------------------------------------------------
WTP <- 500000
v_max_intervention_cost <- WTP * v_incremental_QALY - v_incremental_cost

# 8. CHECKS ----------------------------------------------------------------------
stopifnot(all(abs(sapply(v_wcc_list, sum) - n_cycles) < 1e-8))
cat("WCC weights sum to the number of cycles for every cohort: TRUE\n")

# 9. RESULTS SUMMARY -------------------------------------------------------------
scenario_summary <- data.frame(
  OR = df_or_psd$OR,
  discounted_cost = unname(v_total_discounted_cost_scen[scenarios]),
  discounted_QALY = unname(v_total_discounted_QALY_scen[scenarios]),
  row.names = scenarios
)
cat("\nCost & QALYs by PSD scenario (population-weighted, discounted)\n")
print(transform(scenario_summary, discounted_cost = round(discounted_cost),
                discounted_QALY = round(discounted_QALY, 1)))

comparison_summary <- data.frame(
  incremental_cost = v_incremental_cost,
  incremental_QALY = v_incremental_QALY,
  max_intervention_cost = v_max_intervention_cost
)
cat(sprintf("\nIncremental cost/QALYs and maximum affordable intervention cost (WTP = %s/QALY)\n",
            format(WTP, big.mark = ",", scientific = FALSE)))
print(transform(comparison_summary, incremental_cost = round(incremental_cost),
                incremental_QALY = round(incremental_QALY, 1),
                max_intervention_cost = round(max_intervention_cost)))

# Discounted cost by type, per scenario (population-weighted)
cat("\nDiscounted cost by type (population-weighted)\n")
print(round(t(sapply(l_C_pop, function(m) m["discounted", ]))))

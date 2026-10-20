# Thesis Results Readiness Audit

**Date:** 2026-10-20
**Scope:** read-only audit of the completed ASLA-DP experiments against the approved protocol.
**No experiment was run, no source file was modified.**

**Path note:** the brief names `D:\proposal\Code\Adaptive Single layer deepseek`, which does not
exist on this machine. All artifacts were read from `D:\proposal\Adaptive-Single-layer-deepseek`.
The attack log still records the old path in its header; that is a stale string, not a missing file.

---

## 1. Protocol compliance

| # | Intended condition | Verified actual | Status | Impact |
|---|---|---|---|---|
| 1 | 100 communication rounds per configuration | 62 (`none`), 73 (`clip_only_c2`), 74 (`gaussian_c2_s0.001`), 75 (`gaussian_c2_s0.01`). Stopped by the per-client early-stopping rule (patience 5, min_delta 1e-6; global stop when >=30% of clients have stopped). | Deliberate protocol choice, documented | Arms are compared at different round counts. Must compare at a common round or disclose. |
| 2 | Seed 42, reproducible | `np.random.seed` / `tf.random.set_seed` / `random.seed` are set to 42 at import and to `args.seed` (default 42) in `main()`. The JSON `config` block has **no seed field** and the `.out` logs do not print it. | Traceability gap | Seed cannot be verified from the saved artifacts alone. The reproduction commands in `CHAPTER4_VALIDATED_RESULTS.md` pin `--seed 42`. Disclose. |
| 3 | 10 clients, shared interval, chronological split | 8 clients: AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE. `NI` dropped (ends 2011-01-01, before EKPC starts 2013-06-01). Shared interval 2013-06-01 01:00 .. 2018-08-03 00:00, 45,336 hourly hours. 31,617 train / 13,551 test per client, chronological (`shuffle=False`), `StandardScaler` fitted on train only. | Data-availability limitation, documented | 8 not 10. Disclose wherever the client count is cited. |
| 4 | Model architecture and selected aggregation layer | `Dense(100, relu)` -> `Dense(50, relu)` -> `Dense(1, linear)`, 5 input features. Aggregated layer `model.layers[1]` = `layer2`, 5,050 parameters. | Compliant | - |
| 5 | Local training: 1 epoch, batch 300, Adam 1e-3 | Same in code (`train_local`, `create_model`) and in every JSON `config`. | Compliant | - |
| 6 | Clipping and Gaussian noise applied to the selected-layer update | `apply_dp_gaussian`: flatten the delta -> global L2 clip to C=2.0 -> add N(0, (sigma*C)^2) -> reshape. Delta = local weights minus the received global layer weights, computed after local training and before quantisation/transmission. `clip_only` uses the same clip with no noise draw. | Compliant | - |
| 7 | Metrics, aggregation, test handling | Per-round MAPE = mean over participating clients of each client's Keras MAPE on its own held-out test split, evaluated on the personalised hybrid (global `layer2` + the client's own `layer1`/`layer3`). Final metrics = same evaluation after the last round. Aggregation = unweighted mean of the received `layer2` weights over non-stopped clients. | Compliant, with the B5 caveat | Not one fully global model. Disclose. |
| 8 | Four configurations comparable under one protocol | Same data, split, architecture, optimiser, seed, layer and quantisation. They differ in DP method/sigma, in stopping round, and in the participation schedule (see 2.3). | Comparability limitation | Final-round comparison is confounded by stopping point; common-round comparison is confounded by participation. Disclose; no rerun needed. |

Additional verified details:

- `lookback` is recorded as 24 in the JSON, but `prepare_features` always starts at `max(lookback, 168)` and builds 5 features from lags 1/24/168 plus 24 h and 168 h means. Effective warm-up is 168 h. Identical in all four arms; documentation imprecision only.
- DP diagnostics cover every participating client-round: 563 / 568 / 587 rows = the sum of `history.participants` in each JSON.
- Interpolation touched 49 of 362,688 client-hours (0.0135%), all DST artefacts; no NaN in the assembled matrix.
- All four `.err` files contain only TF/oneDNN startup warnings; no traceback, no non-zero exit.

---

## 2. Results

### 2.1 Final metrics (as reported by each run)

| Config | Rounds | Final MAPE % | Final MAE MW | Best MAPE % (round) |
|---|---:|---:|---:|---:|
| `none` | 62 | 3.0580 | 191.6820 | 3.0450 (40) |
| `clip_only_c2` | 73 | 3.0693 | 192.5119 | 3.0503 (54) |
| `gaussian_c2_s0.001` | 74 | 3.0698 | 192.6104 | 3.0501 (53) |
| `gaussian_c2_s0.01` | 75 | 3.0718 | 192.8672 | 3.0476 (53) |

Spread across arms: 0.0138 pp MAPE, 1.19 MW MAE.

### 2.2 Common-round comparison (round 62, from `history.mape`)

| Config | Participants at r62 | MAPE at r62 % |
|---|---:|---:|
| `none` | 6 | 3.058 |
| `clip_only_c2` | 8 | 3.055 |
| `gaussian_c2_s0.001` | 8 | 3.055 |
| `gaussian_c2_s0.01` | 8 | 3.052 |

At the common round the DP arms are marginally **lower** than `none`, not higher. The final-round
ordering (`none` < `clip_only` < `gaussian 0.001` < `gaussian 0.01`) is therefore not stable across
comparison points.

### 2.3 Convergence and participation

| Round | `none` | `clip_only_c2` | `g0.001` | `g0.01` |
|---:|---:|---:|---:|---:|
| 20 | 4.03 | 15.80 | 15.80 | 15.81 |
| 30 | 3.06 | 4.46 | 4.46 | 4.47 |
| 40 | 3.04 | 3.11 | 3.11 | 3.11 |
| 50 | 3.05 | 3.05 | 3.05 | 3.05 |
| 62 | 3.06 | 3.05 | 3.05 | 3.05 |

`none` keeps all 8 clients for 51 rounds, then 7 for 8 rounds, then 6 for 3. The DP arms keep all 8
for 62 rounds before dropping to 6-7. DP noise slows the local loss improvement, so clients reach the
early-stopping threshold later.

### 2.4 Per-client final MAPE

| Client | `none` | `clip` | `g0.001` | `g0.01` |
|---|---:|---:|---:|---:|
| AEP | 2.58 | 2.59 | 2.59 | 2.60 |
| COMED | 2.84 | 2.87 | 2.87 | 2.88 |
| DAYTON | 3.02 | 3.02 | 3.02 | 3.02 |
| DEOK | 3.24 | 3.24 | 3.24 | 3.25 |
| DOM | 3.44 | 3.46 | 3.46 | 3.46 |
| DUQ | 2.74 | 2.73 | 2.73 | 2.72 |
| EKPC | 4.01 | 4.03 | 4.03 | 4.02 |
| FE | 2.60 | 2.62 | 2.62 | 2.62 |

Largest per-client change between `none` and `gaussian_c2_s0.01`: 0.038 pp.

---

## 3. Privacy evidence and its limits

### 3.1 Mechanism diagnostics (corrected runs)

| Config | Rows | Clip rate overall | Clip rate first 10 | Clip rate last 10 | Mean noise L2 |
|---|---:|---:|---:|---:|---:|
| `clip_only_c2` | 563 | 17.9% | 92.5% | 0.0% | 0.0000 |
| `gaussian_c2_s0.001` | 568 | 17.8% | 92.5% | 0.0% | 0.1422 |
| `gaussian_c2_s0.01` | 587 | 17.4% | 93.8% | 0.0% | 1.4218 |

- Noise L2 scales exactly with sigma: 0.1422 -> 1.4218 for a 10x increase.
- `clip_only_c2` has `noise_l2 = 0.0` in every row, confirming no noise draw.
- Clipping binds only in early rounds. In the last 10 rounds it never binds.
- The raw update norm collapses from 9.606 (first 10 rounds) to 0.0264 (last 10) for
  `gaussian_c2_s0.01`, while the noise norm stays at ~1.42. The median noise-to-raw ratio is 1.60
  (sigma=0.001) and 18.36 (sigma=0.01). In the converged regime the injected noise is therefore much
  larger than the update it perturbs; accuracy is preserved because the model sits in a flat region
  and the noise is averaged over 8 clients.

### 3.2 Attack validation

`privacy_attack_validation_4cfg_r3.txt`:

- 3 rounds, 5 clients (AEP, COMED, DAYTON, DEOK, DOM) + DUQ decoy, 15 cells.
- Data window: 13,896 hours (9,609 train / 4,119 test) - the **pre-correction** loader window, not the
  corrected 45,336-hour window. The file predates the loader rewrite (2026-10-19 10:51 vs
  2026-10-20 09:29).
- Condition 1 (replay correctness) passed: `none` 100% bit-exact; `clip_only_c2` exact to float
  rounding (rel_L2 1.16e-07); both Gaussian arms residual/noise = 1.0000.
- Condition 3 (true series must outscore the decoy in every round) **failed in all four configs**.
  Threat-model margins: -0.0001, -0.0002, -0.0001, -0.0001 (mean). Rounds 2 and 3 are negative in
  every arm.
- The oracle arm shows the signal exists (+0.0118 / +0.0116 / +0.0116 / +0.0098 mean margin), but the
  realistic threat model retains -0.9% to -1.5% of it.
- The identity table (top-1 60.0 / 53.3 / 46.7 / 40.0% vs 20% chance, n=15) is printed, but the gate
  aborted, so it is not a valid ranking result. With 15 cells the confidence interval is wide.

**Conclusion:** the attack study validates the replay pipeline and produces a **negative** result. It
does not demonstrate reconstruction or leakage, and it must not be presented as a successful privacy
evaluation. It also does not apply to the corrected 8-client configuration.

### 3.3 Formal DP

No epsilon/delta accountant exists anywhere in `asla_complete.py` (no `epsilon`, `accountant`, `RDP`
or moments-accounting code). The evidence supports a per-round local mechanism description only.
**No (epsilon, delta) guarantee can be claimed.**

---

## 4. Costs

- No communication or computation cost artifact exists for the corrected runs. `calculate_model_size`
  is defined but only called in the comparison path (line 1035), not in the `--asla-only` path.
- Derivable from implementation evidence, not measured: `layer2` = 5,050 params x 4 bytes = 19.73 KB
  per client-round; 8 clients -> 157.8 KB per round; 62 rounds -> 9.55 MiB (10.02 MB). Local computation =
  1 epoch x ceil(31,617/300) = 106 batches per client-round.
- These are arithmetic consequences of the configuration, not measurements. Any cost claim must be
  labelled as derived.

---

## 5. Supported and unsupported conclusions

### Supported

1. On 8 aligned PJM utilities over 45,336 hourly hours, single-layer ASLA reaches 3.058% MAPE without DP.
2. Adding clipping at C=2.0 and Gaussian noise at sigma=0.001 or 0.01 does not materially change final
   accuracy: the largest per-client change is 0.038 pp and the arm-level spread is 0.014 pp.
3. DP slows convergence: at round 20 the DP arms are at 15.8% vs 4.03% without DP; they reach ~3.05%
   by round 40-50.
4. The mechanism is active and measured: clipping binds in early rounds, noise L2 scales exactly with
   sigma, and `clip_only` draws no noise.
5. No formal DP guarantee is established, and no usable leakage evidence exists.

### Unsupported - must not be claimed

1. Any (epsilon, delta) guarantee, or any inference of epsilon from sigma.
2. That the attack demonstrates privacy or leakage.
3. A precise causal attribution of the final-round difference to clipping rather than noise. The
   final-round comparison is confounded by different stopping points; at round 62 the DP arms are
   marginally better.
4. Any measured communication or computation cost.
5. Statistical significance of the arm ordering. The runs are single-seed, and the only available
   variability estimate (0.05-0.08 pp) comes from the pre-correction configuration.
6. That the privacy study applies to the corrected 8-client configuration.

---

## 6. Verdict

**A - the existing results are sufficient to begin final thesis writing, provided the limitations in
sections 1 and 5 are disclosed.**

Reasons:

- The forecasting comparison is on a defensible, aligned, 8-client dataset and the accuracy
  conclusion is robust to the comparison point.
- The privacy result is a legitimate negative result: the gate aborted, and the thesis can say so.
- The remaining issues are disclosure and wording, not missing measurements.

---

## 7. Next action

**None. No further experiment is required before writing.**

The only text corrections needed:

- Replace "DP cost = +0.011 to +0.014 pp, dominated by clipping" with "no measurable accuracy cost at
  convergence; DP slows convergence".
- State that the attack study used a 5-client, 3-round, pre-correction window and aborted.
- State that no (epsilon, delta) guarantee is claimed.
- State 8 clients, not 10.

If a reviewer later requires a privacy claim on the corrected configuration, the minimum action would
be to rerun `asla_privacy_attack.py` on the corrected 8-client window. That is not required for the
approved scope.

---

## 8. Artifact index

| Artifact | Supports |
|---|---|
| `corrected_100r_none.{txt,json,out,err}` | No-DP baseline: 62 rounds, 3.0580% MAPE, 191.6820 MW MAE |
| `corrected_100r_clip_only_c2.{txt,json,out,err}` | Clip-only ablation: 73 rounds, 3.0693% MAPE |
| `corrected_100r_gaussian_c2_s0.001.{txt,json,out,err}` | Gaussian sigma=0.001: 74 rounds, 3.0698% MAPE |
| `corrected_100r_gaussian_c2_s0.01.{txt,json,out,err}` | Gaussian sigma=0.01: 75 rounds, 3.0718% MAPE |
| `corrected_dp_diag_clip_only_c2.csv` | 563 client-rounds, noise_l2 = 0, clip rate 17.9% |
| `corrected_dp_diag_gaussian_c2_s0.001.csv` | 568 client-rounds, noise L2 0.1422 |
| `corrected_dp_diag_gaussian_c2_s0.01.csv` | 587 client-rounds, noise L2 1.4218 |
| `privacy_attack_validation_4cfg_r3.txt` | Attack gate ABORT in all four configs; replay condition 1 passed |
| `CHAPTER4_VALIDATED_RESULTS.md` | Prior write-up; its "dominated by clipping" wording needs the correction in section 7 |
| `AUDIT_REPORT.md` | Pre-correction defects D1/D2, F5 (attack aborts), F11 (no epsilon accounting) |
| `asla_complete.py` | Loader whitelist, `apply_dp_gaussian`, `apply_dp_clip_only`, early stopping, metrics |

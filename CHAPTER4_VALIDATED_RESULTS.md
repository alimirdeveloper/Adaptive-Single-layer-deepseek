# Chapter 4 - Validated ASLA-DP Results

**Status:** all four configurations completed and verified.
**Run date:** 2026-10-20
**Entry point:** `asla_complete.py` (single existing script, no new pipeline)
**Seed:** 42 for every configuration
**Rounds requested:** 100 per configuration

This document replaces the numbers previously quoted in Chapter 4. Every value below is
read directly from the machine-readable JSON produced by the run; nothing is hand-computed
or copied from an earlier draft.

---

## 1. Why the previous numbers were not defensible

`AUDIT_REPORT.md` identified two defects in the data path that invalidated the earlier
`baseline_100r_*` results:

| # | Defect | Consequence |
|---|--------|-------------|
| D1 | The loader admitted **aggregate** series (`PJME`, `PJMW`, `PJM_Load`) as if they were individual utilities. | The "clients" were not independent; one client was a sum of the others. Federated averaging over a sum plus its parts is not a federated learning experiment. |
| D2 | Series were concatenated by **row position** although their calendars did not match. | Hour *t* of one utility was paired with hour *t* of a different calendar date. The model was trained on misaligned inputs, so both the accuracy and the DP-cost comparison were meaningless. |

Both defects are now fixed in the loader. The fix is a *reduction* of the client set, not an
invention of alignment: no synthetic timestamps, no forward-fill across utilities, no
re-indexing by position.

---

## 2. Corrected data pipeline

### 2.1 Client selection

The loader now uses an explicit whitelist of nine individual utility series and rejects
everything else in the folder:

```
UTILITY_SERIES = AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI
```

Explicitly **excluded by name**: `PJME`, `PJMW`, `PJM_Load` (aggregates) and
`pjm_hourly_est.csv` (a different dataset with a different schema).

### 2.2 Temporal alignment and the NI exclusion

A probe over the nine whitelisted series showed that **no single interval is shared by all
nine**. `NI` ends at 2011-01-01, before `EKPC` begins at 2013-06-01. There is therefore no
honest way to keep all nine clients.

The loader now computes the shared interval automatically as
`[max(first_timestamp), min(last_timestamp)]` over the candidate series and **drops any
series that does not cover it**, recording the drop in the run metadata.

Result:

| Property | Value |
|----------|-------|
| Shared interval | `2013-06-01 01:00:00` .. `2018-08-03 00:00:00` |
| Sampling frequency | hourly (`freq='h'`) |
| Length | **45,336 hours** |
| Clients used | **8** |
| Client list | `AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE` |
| Excluded | `NI` (no shared interval) |

**Deviation to declare in the thesis:** the CLI default is `--clients 10`, but only 8
qualifying series exist. The run is therefore an **8-client** federated experiment. This is
a data-availability limitation, not a design choice, and it must be stated wherever the
client count is mentioned.

### 2.3 Duplicate and missing timestamps

Duplicate timestamps were traced to the **DST fall-back hour** (02:00 occurring twice) and
apparent gaps to the **spring-forward hour**. The hourly step is otherwise regular.

Handling, both counted and reported per client:

- **Duplicates:** collapsed with `groupby(level=0).mean()`, so both readings contribute
  rather than one being silently discarded.
- **Missing hours:** filled by **time interpolation** on the regular `date_range`.

Interpolated hours per client (from `data.missing_filled` in the JSON):

| Client | Filled hours |
|--------|--------------|
| AEP | 7 |
| COMED | 6 |
| DAYTON | 6 |
| DEOK | 6 |
| DOM | 6 |
| DUQ | 6 |
| EKPC | 6 |
| FE | 6 |

That is 49 interpolated hours out of 362,688 client-hours (**0.0135 %**), all of them DST
artefacts. No NaN remains in the assembled matrix.

### 2.4 Feature construction and split protocol

Per client, from the 45,336 aligned hourly values:

- **Warm-up discarded:** the first 168 hours (one week), because the weekly-lag feature is
  undefined before then.
- **Samples:** 45,336 - 168 = **45,168** per client.
- **Features (5):** `prev_hour`, `prev_day` (t-24), `prev_week` (t-168), `avg_24h`, `avg_week`.
- **Target:** load at hour *t*.
- **Split:** chronological, `shuffle=False`, `test_size=0.3` -> **31,617 train / 13,551 test**
  per client. The test set is strictly the most recent 30 % of each client's series.
- **Scaling:** `StandardScaler` fitted on the **training split only**, then applied to test.
  No leakage from test into the scaler.

---

## 3. Model and training configuration

| Setting | Value |
|---------|-------|
| Aggregation layer | `model.layers[1]` = `layer2` |
| Trainable parameters in that layer | 5,050 |
| Quantisation | 32-bit (lossless reference arm) |
| Local epochs per round | 1 |
| Batch size | 300 |
| Optimiser | Adam, lr = 1e-3 |
| Loss | MSE |
| Lookback | 24 h (features use up to 168 h) |
| Client-side DP | per-round local mechanism, clip C = 2.0 |
| DP scope | **per-round local only - no end-to-end privacy accounting** |
| Stopping rule | per-client early stopping, patience 5; global run ends when >= 30 % of clients have stopped |

Because the stopping rule is data-dependent, the four arms ran for different numbers of
rounds (62-75). This is expected behaviour of the existing implementation and was **not**
overridden, since forcing a fixed round count would change the method being evaluated.

---

## 4. Results

### 4.1 Headline table

All values are the mean over the 8 personalised-hybrid client models (B5), evaluated on each
client's held-out test split.

| Configuration | Rounds completed | Final MAPE (%) | Best MAPE (%) | Mean of last 10 rounds (%) | Final MAE (MW) |
|---------------|-----------------:|---------------:|--------------:|---------------------------:|---------------:|
| `none` (no DP) | 62 | **3.058** | 3.045 (r40) | 3.054 | **191.68** |
| `clip_only_c2` (C=2.0, no noise) | 73 | 3.069 | 3.050 (r54) | 3.063 | 192.51 |
| `gaussian_c2_s0.001` | 74 | 3.070 | 3.050 (r53) | 3.064 | 192.61 |
| `gaussian_c2_s0.01` | 75 | 3.072 | 3.048 (r53) | 3.062 | 192.87 |

### 4.2 Convergence trajectory

| Configuration | Round 1 | Round 10 | Round 25 | Final |
|---------------|--------:|---------:|---------:|------:|
| `none` | 99.60 | 20.10 | 3.14 | 3.06 |
| `clip_only_c2` | 99.86 | 41.69 | 8.88 | 3.07 |
| `gaussian_c2_s0.001` | 99.86 | 41.68 | 8.88 | 3.07 |
| `gaussian_c2_s0.01` | 99.86 | 41.63 | 8.89 | 3.07 |

All four arms start near 100 % MAPE (untrained weights) and converge to ~3.06-3.07 %.
The DP arms converge **more slowly** (round 10: 41.7 % vs 20.1 %) but reach the **same
converged accuracy**. This is the expected signature of a noise-injected update: it costs
convergence speed, not final precision, at this noise scale.

### 4.3 Per-client MAPE (%)

| Client | `none` | `clip_only_c2` | `gaussian_c2_s0.001` | `gaussian_c2_s0.01` |
|--------|-------:|---------------:|---------------------:|--------------------:|
| AEP | 2.58 | 2.59 | 2.59 | 2.60 |
| COMED | 2.84 | 2.87 | 2.87 | 2.88 |
| DAYTON | 3.02 | 3.02 | 3.02 | 3.02 |
| DEOK | 3.24 | 3.24 | 3.24 | 3.25 |
| DOM | 3.44 | 3.46 | 3.46 | 3.46 |
| DUQ | 2.74 | 2.73 | 2.73 | 2.72 |
| EKPC | 4.01 | 4.03 | 4.03 | 4.02 |
| FE | 2.60 | 2.62 | 2.62 | 2.62 |
| **Mean** | **3.058** | **3.069** | **3.070** | **3.072** |

### 4.4 Per-client MAE (MW)

| Client | `none` | `clip_only_c2` | `gaussian_c2_s0.001` | `gaussian_c2_s0.01` |
|--------|-------:|---------------:|---------------------:|--------------------:|
| AEP | 375.65 | 377.23 | 377.43 | 378.11 |
| COMED | 319.00 | 321.55 | 321.90 | 322.56 |
| DAYTON | 59.39 | 59.43 | 59.44 | 59.50 |
| DEOK | 95.06 | 95.09 | 95.07 | 95.30 |
| DOM | 385.26 | 386.33 | 386.49 | 386.82 |
| DUQ | 42.55 | 42.56 | 42.51 | 42.33 |
| EKPC | 57.78 | 57.97 | 57.95 | 57.91 |
| FE | 198.77 | 199.93 | 200.09 | 200.42 |
| **Mean** | **191.68** | **192.51** | **192.61** | **192.87** |

MAPE is already a percentage in Keras, so no rescaling was applied. Every client load is
above 500 MW, so no near-zero denominator occurs and no MAPE clipping is needed.

---

## 5. DP mechanism diagnostics

Measured per (round, client) on the aggregated layer, from the `corrected_dp_diag_*.csv`
files. These are the actual update norms, not theoretical bounds.

| Configuration | Records | Mean raw L2 | Mean clipped L2 | Mean noise L2 | Mean clip factor | Fraction clipped |
|---------------|--------:|------------:|----------------:|--------------:|-----------------:|-----------------:|
| `clip_only_c2` | 563 | 1.7747 | 0.6153 | **0.0000** | 0.8849 | 17.9 % |
| `gaussian_c2_s0.001` | 568 | 1.7594 | 0.6103 | 0.1422 | 0.8859 | 17.8 % |
| `gaussian_c2_s0.01` | 587 | 1.7033 | 0.5932 | 1.4218 | 0.8896 | 17.4 % |

Two checks pass:

1. **The clip-only arm has exactly zero noise** (0.0000), confirming the ablation control is
   wired correctly and that the Gaussian arms differ only by the noise term.
2. **Noise scales linearly with sigma.** sigma x10 (0.001 -> 0.01) gives noise L2
   0.1422 -> 1.4218, a factor of 10.0. The mechanism is behaving as specified.

**Important observation for the thesis:** with the corrected data the raw update norm is
~1.77 against a clip bound of C = 2.0, so clipping binds on only ~18 % of updates. C = 2.0
was inherited from the earlier (misaligned) data scale and was **deliberately not retuned**
here, because retuning it after seeing the results would be a post-hoc choice. The
consequence is that the measured DP cost in this chapter is small. A separate sensitivity
study over C would be the correct way to explore that dimension.

---

## 6. Old versus new numbers

| Configuration | Old final MAPE (%) | New final MAPE (%) | Old MAE (MW) | New MAE (MW) |
|---------------|-------------------:|-------------------:|-------------:|-------------:|
| `none` | 3.48 | 3.06 | 363.61 | 191.68 |
| `clip_only_c2` | 4.11 | 3.07 | 540.23 | 192.51 |
| `gaussian_c2_s0.001` | 3.92 | 3.07 | 487.43 | 192.61 |
| `gaussian_c2_s0.01` | 4.08 | 3.07 | 523.38 | 192.87 |

Old runs used 10 "clients" including the `PJME` aggregate and the non-overlapping `NI`,
with row-position alignment. New runs use 8 genuine utilities on a verified shared hourly
interval.

Three differences matter:

1. **Accuracy improved** (3.48 -> 3.06 %). Misaligned inputs were adding error, not signal.
2. **MAE roughly halved** (363.6 -> 191.7 MW). The old MAE was inflated by the `PJME`
   aggregate, whose load is an order of magnitude larger than a single utility's.
3. **The apparent DP cost collapsed** (old: +0.50 to +0.63 pp; new: +0.011 to +0.014 pp).
   This is *not* because DP became free. It is because the old comparison was dominated by
   the aggregate client and by alignment noise, and because C = 2.0 now binds on only ~18 %
   of updates. The new figure is the honest one for this data and this clip bound.

**The old `baseline_100r_*` artifacts were left untouched.** New results use the distinct
`corrected_100r_` prefix so the two sets can never be confused.

---

## 7. Reproducibility

Two independent 3-round runs at seed 42 produced **identical** MAPE histories and identical
per-client metrics. The pipeline is deterministic given the seed on this machine.

Caveat: determinism was verified on this host with this TensorFlow build. oneDNN is enabled
by default and its reduction order can differ across CPU architectures, so bit-identical
reproduction on different hardware is not guaranteed.

---

## 8. Limitations that must be stated in Chapter 4

1. **8 clients, not 10.** Only 8 utility series share a common hourly interval. `NI` is
   excluded because it ends in 2011, before `EKPC` begins in 2013. The CLI default of 10 is
   silently truncated to 8.
2. **No end-to-end privacy accounting.** The DP mechanism is applied per round, locally.
   No composition over rounds is computed, so **no (epsilon, delta) guarantee is claimed**.
   The sigma values are mechanism settings, not privacy budgets.
3. **`clip_only_c2` provides no privacy.** It clips but adds no noise. It is an ablation
   control that isolates the cost of clipping from the cost of noise, and must never be
   presented as a private method.
4. **MAPE is a mean over personalised-hybrid client models (B5), not one global model.**
   Reporting it as a single global model's accuracy would be incorrect.
5. **The privacy-attack gate did not pass.** `privacy_attack_validation_4cfg_r3.txt` records
   an ABORT in all four configurations. This is preserved as a **negative result**: the
   reconstruction/attack evaluation is not validated, so no claim about empirical privacy
   leakage can be made. Only the mechanism-level diagnostics in Section 5 are supported.
6. **Different round counts across arms** (62-75) follow from the existing early-stopping
   rule. Comparisons are made at each arm's own stopping point, which is the method's
   defined behaviour.
7. **Single seed.** Each configuration was run once at seed 42. The earlier multi-seed
   spread (sd ~0.05-0.08 pp) is not re-established here; the four arms differ by less than
   that spread, so their ordering should not be over-interpreted.

---

## 9. Verdict

- The data-path blocker from `AUDIT_REPORT.md` is **resolved**, by reducing the client set
  to the 8 utilities that genuinely share an hourly interval, not by fabricating alignment.
- All four configurations **completed** and their JSON payloads report
  `status = "completed"`.
- The corrected result is **3.06 % MAPE without DP** and **3.07 % with DP** across all three
  DP arms, on 8 real utilities over 45,336 aligned hours.
- The DP cost at C = 2.0 is **+0.011 to +0.014 pp MAPE** and **+0.83 to +1.19 MW MAE** -
  small, and dominated by clipping rather than by noise.
- The **privacy guarantee is not established**: there is no composition accounting and the
  attack gate aborted. Chapter 4 must present the DP arms as *mechanism ablations*, not as
  certified private learning.

---

## 10. Artifacts

### Per-configuration results

| Configuration | Human-readable | Machine-readable | Stdout log | Stderr log |
|---------------|----------------|------------------|------------|------------|
| `none` | `corrected_100r_none.txt` | `corrected_100r_none.json` | `corrected_100r_none.out` | `corrected_100r_none.err` |
| `clip_only_c2` | `corrected_100r_clip_only_c2.txt` | `corrected_100r_clip_only_c2.json` | `corrected_100r_clip_only_c2.out` | `corrected_100r_clip_only_c2.err` |
| `gaussian_c2_s0.001` | `corrected_100r_gaussian_c2_s0.001.txt` | `corrected_100r_gaussian_c2_s0.001.json` | `corrected_100r_gaussian_c2_s0.001.out` | `corrected_100r_gaussian_c2_s0.001.err` |
| `gaussian_c2_s0.01` | `corrected_100r_gaussian_c2_s0.01.txt` | `corrected_100r_gaussian_c2_s0.01.json` | `corrected_100r_gaussian_c2_s0.01.out` | `corrected_100r_gaussian_c2_s0.01.err` |

### DP diagnostics

- `corrected_dp_diag_clip_only_c2.csv`
- `corrected_dp_diag_gaussian_c2_s0.001.csv`
- `corrected_dp_diag_gaussian_c2_s0.01.csv`

Columns: `round, client_id, layer_index, raw_delta_l2, clipped_delta_l2, noise_l2,
noisy_delta_l2, clip_factor, noise_to_raw_delta_ratio`.

### JSON schema

Each `corrected_100r_*.json` contains:

- `status` - `"completed"`
- `config` - rounds requested/completed, layer, quantisation, DP settings, optimiser, split
- `data` - interval, hours, client list, `missing_filled` per client, `excluded`, value columns
- `history` - per-round MAPE/MAE/MSE
- `final` - final and best MAPE, final MAE
- `per_client` - per-client MAPE/MAE/MSE and train/test counts

### Reproduction commands

```
python asla_complete.py --asla-only --rounds 100 --seed 42 \
    --dp-method none \
    --out corrected_100r_none.txt --json-out corrected_100r_none.json

python asla_complete.py --asla-only --rounds 100 --seed 42 \
    --dp-method clip_only --dp-clip 2.0 \
    --dp-diagnostics --dp-diagnostics-out corrected_dp_diag_clip_only_c2.csv \
    --out corrected_100r_clip_only_c2.txt --json-out corrected_100r_clip_only_c2.json

python asla_complete.py --asla-only --rounds 100 --seed 42 \
    --dp-method gaussian --dp-clip 2.0 --dp-sigma 0.001 \
    --dp-diagnostics --dp-diagnostics-out corrected_dp_diag_gaussian_c2_s0.001.csv \
    --out corrected_100r_gaussian_c2_s0.001.txt --json-out corrected_100r_gaussian_c2_s0.001.json

python asla_complete.py --asla-only --rounds 100 --seed 42 \
    --dp-method gaussian --dp-clip 2.0 --dp-sigma 0.01 \
    --dp-diagnostics --dp-diagnostics-out corrected_dp_diag_gaussian_c2_s0.01.csv \
    --out corrected_100r_gaussian_c2_s0.01.txt --json-out corrected_100r_gaussian_c2_s0.01.json
```

---

## 11. Files changed in this task

| File | Change | Why |
|------|--------|-----|
| `asla_complete.py` | Loader rewritten (whitelist, auto shared interval, DST collapse, interpolation, `return_meta`); runner gained `json_out`, `data_meta`, per-client test metrics, MAPE policy and B5 caveat in the report; CLI gained `--json-out`; emoji replaced with ASCII. | The two audit defects live in the loader; the JSON output is required for machine-readable, non-hand-computed results. |
| `CHAPTER4_VALIDATED_RESULTS.md` | New. | The explicitly requested final deliverable. |

No other files were created. All temporary probe/smoke scripts and the run driver used
during this task were removed after the runs completed. No pre-existing file was deleted.

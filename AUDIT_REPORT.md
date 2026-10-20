# Audit report — Chapters ۳ and ۴ of `پروپوزال.docx` vs. the experiment artifacts

**Date:** 2026-10-19
**Document audited:** `D:\proposal\master-degree-documentry\پروپوزال.docx`
**Code audited:** `asla_complete.py`, `asla_privacy_attack.py`, `run_fedavg_only.py`
**Artifacts audited:** `baseline_*.txt`, `baseline_100r_*.out`, `privacy_attack_*_4cfg_r3.{csv,txt}`

---

## 0. Scope and method

The `.docx` was read without Office automation: a `.docx` is a ZIP, so `word/document.xml`
and `word/styles.xml` were parsed directly with the standard library (`_docx_dump.py`).
This gives paragraph-index-level access to text and to the applied paragraph style, which is
what the structural findings below rest on.

Every number in the document was then traced to a file in this repository. A number is
recorded as **supported** only if it can be regenerated from a committed artifact; otherwise
it is recorded as **unsupported**, with the closest artifact value given for comparison.

Reproducibility was tested empirically rather than assumed (`_seed_check.py`): two
`create_model()` calls were made in one process under `np.random.seed` + `tf.random.set_seed`
only, and the initial-weight digests were compared.

---

## 1. Findings, by severity

| # | Severity | Finding |
|---|----------|---------|
| F1 | **Blocker** | Chapter ۴ reports MAPE ≈ 11–12 % for the no-DP method; every artifact in the repo reports **3.39–3.49 %**. No round budget reproduces 11.29 %. |
| F2 | **Blocker** | `FedAvg = 78.52 %` (table ۴-۴) has **no artifact at all** — `run_fedavg_only.py` exists but no output file does, and no file in the repo contains `78.5`. |
| F3 | **Blocker** | The layer sweep (table ۴-۲: 12.24 / 11.29 / 13.60) has **no artifact**; the code has no layer-sweep mode. |
| F4 | **Blocker** | Table ۴-۱ is **not reproducible from the shipped CSVs under any single window** — its `min` matches the analysis window while its `max` matches the full series. |
| F5 | **Blocker** | The privacy attack **does not demonstrate an attack**. The feasibility gate ABORTs in all four configs at the realistic knowledge level. |
| F6 | High | `--seed` did not control Keras 3 initial weights, so the seed-42 baselines were **not reproducible**. *Fixed in this session.* |
| F7 | High | The loader treats **aggregate and duplicate CSVs as clients**: 13 files pass its filter, but only 9 are distinct utilities. |
| F8 | High | The loader globs **every** `*.csv` in the folder, so result files can silently become "clients"; it also crashes on a cp1252 console (`✓`). |
| F9 | High | The 10 "clients" are **not time-aligned**: their windows span 2002-01 to 2015-01. |
| F10 | Medium | Document structure: ۴-۳ and ۴-۳-۱ are styled `Normal`; ۳-۳ is `heading 2` among `heading 3` siblings; tables ۴-۱…۴-۵ are box-drawing art, not Word tables; the conclusion is **truncated mid-sentence**. |
| F11 | Medium | ε is reported in tables ۴-۴/۴-۵ but the code parameterises **σ**, never ε; no privacy accounting exists. |
| F12 | Low | `clip_only` is presented as a privacy mechanism; it is not one. |

---

## 2. F1 — The chapter-۴ numbers are on a different scale from the artifacts

The document's table ۴-۴ reads:

| روش | MAPE (%) | ε میانگین | حجم تبادل (KB) |
|---|---|---|---|
| ASLA (بدون DP) | ۱۱.۲۹ | ∞ | ۷,۸۹۰ |
| FedAvg | ۷۸.۵۲ | ∞ | ۸,۹۵۸ |
| DP-FedAvg | ۱۴.۲۰ | ۱.۰ | ۸,۹۵۸ |
| DP-ASLA (ما) | ۱۱.۰۷ | ۱.۰۷ | ۷,۸۹۰ |

The repository's own 100-round baselines, four seeds each, are:

| config | seed 42 | 101 | 202 | 303 | mean ± sd | mean MAE |
|---|---|---|---|---|---|---|
| `none` (no DP) | 3.48 | 3.49 | 3.39 | 3.49 | **3.46 ± 0.05** | 363.0 |
| `clip_only_c2` | 4.11 | 4.07 | 4.01 | 3.93 | **4.03 ± 0.08** | 518.4 |
| `gaussian_c2_s0.001` | 3.92 | 4.08 | 3.93 | 3.91 | **3.96 ± 0.08** | 506.8 |
| `gaussian_c2_s0.01` | 4.08 | 4.00 | 4.04 | 3.93 | **4.01 ± 0.06** | 513.3 |

The no-DP method converges to **3.46 %**, not 11.29 %. The gap is a factor of 3.3.

The obvious defence — "the document used fewer rounds" — does not hold. The per-round trace
of `baseline_100r_none.out` is:

```
Round  1: 99.97%    Round 15: 47.59%    Round 30: 22.51%
Round  5: 98.06%    Round 20: 34.33%    Round 50: 10.29%
Round 10: 72.61%    Round 25: 27.18%    Round 100: 3.48%
```

At round 20 the method is at **34.33 %**, not 11.29 %. The value 11.29 % is bracketed by
round 47 (11.63 %) and round 48 (11.16 %) — but a single round index cannot simultaneously
explain 12.24, 13.60, 14.20, 11.5, 11.2 and 12.0, which the document reports for *different
architectures and different α*. **No round budget, seed, or config in this repository
produces the chapter-۴ number set.**

**Consequence:** the chapter's central claim ("DP-ASLA reaches 11.07 % at ε = 1.07") is not
supported by the code that is supposed to produce it. Either the numbers come from a run
whose artifacts were never saved, or they were transcribed from elsewhere. Both must be
resolved before submission.

---

## 3. F2 — `FedAvg = 78.52 %` has no provenance

`run_fedavg_only.py` is committed, and four PNGs mention FedAvg
(`asla_vs_fedavg.png`, `fedavg_daily_results.png`, `fedavg_no_early_stop.png`,
`fedavg_only_results.png`), but:

* no `*.txt`, `*.csv`, `*.out` or `*.json` in the repository contains `78.5`;
* no log of a FedAvg run exists.

A 78.52 % MAPE for FedAvg against a 3.46 % centralised baseline is also implausible on its
face: it would mean the federated model is 23× worse than the same model trained centrally,
which is not what the ASLA-vs-FedAvg figure in the repo shows. This number should be
regenerated or removed.

---

## 4. F3 — The layer sweep has no artifact

Table ۴-۲ reports 12.24 / 11.29 / 13.60 % for 1 / 2 / 3 layers. `asla_complete.py` has no
layer-sweep flag, and no artifact contains those three values. The middle value (11.29) is
identical to the no-DP row of table ۴-۴, which suggests the sweep was not actually run as a
separate experiment.

---

## 5. F4 — Table ۴-۱ cannot be reproduced from the data

Document values vs. the loader's actual analysis window (first 13,896 rows after the
ascending sort):

| client | doc mean | window mean | doc sd | window sd | doc range | window range |
|---|---|---|---|---|---|---|
| AEP | 16,067 | **15,735** | 2,549 | **2,399** | 9,823–25,695 | **9,823–24,015** |
| COMED | 11,521 | **11,734** | 2,399 | **2,623** | 7,237–21,753 | **7,382–23,753** |
| DAYTON | 2,126 | **2,097** | 405 | **379** | 982–3,746 | **1,202–3,722** |

Four candidate windows were tested for each client (full series, first 13,896, last 13,896,
middle 13,896). **None matches the document on all four statistics.** The pattern of the
mismatch is diagnostic:

* AEP `min = 9,823` is the **window** minimum, but AEP `max = 25,695` is the **full-series**
  maximum (the window maximum is 24,015).
* COMED `min = 7,237` is the **full-series** minimum (the window minimum is 7,382), and
  `max = 21,753` matches nothing — the full series is 23,753, i.e. a digit transposition.
* DAYTON `min = 982` and `max = 3,746` are both **full-series** extremes.

So the table mixes statistics from two different slices and contains at least one
transcription error. It must be recomputed from a single, stated window.

---

## 6. F5 — The privacy attack is a negative result, and the document must say so

`privacy_attack_validation_4cfg_r3.txt` runs a three-condition feasibility gate before any
ranking stage is allowed. The outcome, for all four configs:

```
config                 oracle exact  oracle cos   B cos      decoy cos   margin    separation
none                   100%          1.000000     0.912685   0.912790    -0.0001   0.1578
clip_only_c2           ~exact        1.000000     0.911918   0.912078    -0.0002   0.1619
gaussian_c2_s0.001     n/a (noise)   0.997482     0.909403   0.909520    -0.0001   0.1606
gaussian_c2_s0.01      n/a (noise)   0.815014     0.744264   0.744407    -0.0001   0.1089
```

Condition 3 fails everywhere: the true series does **not** outscore the decoy control in
every round. The gate therefore **ABORTs** in all four configs:

> `ABORT: the true series does not outscore the decoy control in every round ... round(s) 2, 3 carry no series-specific information, so a ranking would be meaningless there.`

The diagnosis is precise and worth keeping in the thesis: the signal *exists* — with the
pre-round state known (oracle arm) the true series beats the decoy by **+0.0118** (none),
+0.0116 (clip), +0.0116 (σ=0.001), +0.0098 (σ=0.01) — but at the realistic knowledge level
the unmodelled pre-round state buries it, leaving **−0.9 % / −1.4 % / −1.0 % / −1.5 %** of
the ceiling.

**What this supports:** "we built a rigorous test for whether an identity-inference attack is
feasible, and it correctly refused to proceed; the attack is not feasible under this threat
model, and we can say exactly why."
**What it does not support:** any claim that an attack succeeded, or any ε-vs-attack-accuracy
curve.

Note also that the oracle cosine for σ=0.01 is **0.815**, not ~1.0 — the replay is only
approximate at that noise level, which is expected but should be stated.

---

## 7. F6 — `--seed` was not reproducible (fixed)

Keras 3's seedless initializers (`GlorotUniform`, `Orthogonal`, …) draw from Python's
`random` module, not from the TensorFlow global RNG. Seeding `numpy` and `tensorflow` alone
therefore leaves initial weights unseeded. Verified empirically: two `create_model()` builds
in one process produced **different** weight digests under `np`+`tf` seeding, and **identical**
digests after adding `random.seed`.

Fix applied to `asla_complete.py`:

* `import random` added;
* `random.seed(42)` at module level (so imports and helper calls are deterministic);
* `random.seed(args.seed)` in `main()`, before any model is built;
* the stale B6 docstring corrected (`os.listdir` → sorted CSV list).

`py_compile` and a clean import both pass.

**Caveat that must be stated in the thesis:** the four `baseline_*_seed101/202/303` runs and
the seed-42 runs were produced **before** this fix. They are valid as four independent
initialisations, but they are **not bit-reproducible**, and the seed-42 run cannot be
regenerated to the same weights. If the thesis claims reproducibility, the baselines must be
re-run after the fix.

---

## 8. F7 / F8 — The loader's client set is contaminated

`load_pjm_multiple_files` globs `*.csv` and accepts the first numeric column with more than
1,000 rows. On the current folder, **13 files pass**:

| file | rows | verdict |
|---|---|---|
| AEP_hourly.csv | 121,273 | client via `AEP_MW` |
| COMED_hourly.csv | 66,497 | client via `COMED_MW` |
| DAYTON_hourly.csv | 121,275 | client via `DAYTON_MW` |
| DEOK_hourly.csv | 57,739 | client via `DEOK_MW` |
| DOM_hourly.csv | 116,189 | client via `DOM_MW` |
| DUQ_hourly.csv | 119,068 | client via `DUQ_MW` |
| EKPC_hourly.csv | 45,334 | client via `EKPC_MW` |
| FE_hourly.csv | 62,874 | client via `FE_MW` |
| NI_hourly.csv | 58,450 | client via `NI_MW` |
| PJME_hourly.csv | 145,366 | client via `PJME_MW` |
| PJMW_hourly.csv | 143,206 | client via `PJMW_MW` |
| PJM_Load_hourly.csv | 32,896 | client via `PJM_Load_MW` |
| **pjm_hourly_est.csv** | 178,262 | client via **`AEP`** — a duplicate of AEP |

Only **9** are distinct utilities. `PJME`, `PJMW` and `PJM_Load` are **aggregate zones**, not
clients, and `pjm_hourly_est.csv` is a wide table whose first numeric column is `AEP`, i.e. a
second copy of the same series. The code then takes the first 10 sorted names, so the actual
client set is:

```
AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME
```

— nine utilities plus one aggregate. The document's "۱۳ مشترک" is therefore wrong twice
over: the loader's 13 includes duplicates and aggregates, and the code uses 10.

The aligned window is `min(values)` capped at 13,896, so the analysis uses **13,896 samples
per client**, not the "بیش از ۳۲,۰۰۰ نمونه" the document claims. (32,896 is `PJM_Load`'s row
count — the pre-cap minimum, which is where the document's figure appears to come from.)

Because the glob is unfiltered, any `privacy_attack_*.csv` with more than 1,000 rows would
also become a client. Today they are all small enough to be skipped, but that is luck, not
design. **Recommended fix:** whitelist `*_hourly.csv` and exclude `privacy_attack_*`,
`pa_*` and `pjm_hourly_est.csv`.

Separately, the loader prints `✓`, which raises `UnicodeEncodeError` on a cp1252 console and
kills the run before training starts. This is why every command in this session sets
`PYTHONIOENCODING=utf-8`.

---

## 9. F9 — The clients are not time-aligned

Each client's window is the **oldest** 13,896 hours of its own series, so the ten clients
cover ten different calendar periods:

| client | window start | window end |
|---|---|---|
| PJME | 2002-01-01 | 2003-08-03 |
| NI | 2004-05-01 | 2005-12-01 |
| AEP | 2004-10-01 | 2006-05-03 |
| DAYTON | 2004-10-01 | 2006-05-03 |
| DUQ | 2005-01-01 | 2006-08-03 |
| DOM | 2005-05-01 | 2006-12-01 |
| COMED | 2011-01-01 | 2012-08-02 |
| FE | 2011-06-01 | 2012-12-31 |
| DEOK | 2012-01-01 | 2013-08-02 |
| EKPC | 2013-06-01 | 2015-01-01 |

Federated averaging therefore mixes 2002 load levels with 2014 load levels, and each
client's test split falls in a different year. This is a genuine methodological weakness that
a reviewer will find; it should either be fixed (intersect the date ranges, or use a common
recent window) or disclosed.

---

## 10. F10 — Document structure defects

From `word/document.xml` (905 paragraphs, 5 real `<w:tbl>`, 0 drawings):

1. **۴-۳ is missing as a heading.** The heading sequence jumps ۴-۱, ۴-۲, ۴-۴, ۴-۵, ۴-۶, ۴-۷.
   The paragraphs that should be ۴-۳ and ۴-۳-۱ are styled **`Normal`**, so they do not appear
   in the table of contents and are not navigable.
2. **۳-۳ is styled `heading 2`** while its siblings ۳-۱ and ۳-۲ are `heading 3`. It will
   appear at the wrong level in the TOC.
3. **Tables ۴-۱ … ۴-۵ are box-drawing art**, not Word tables — e.g. `┌───┬───┐` / `│` / `└───┴───┘`
   typed as body text. They cannot be numbered, captioned, sorted, or read by a screen
   reader, and they will reflow badly. The document does contain 5 real tables elsewhere, so
   the capability exists; these five were simply not converted.
4. **The conclusion is truncated mid-sentence.** Paragraph 800 ends with
   `... روش پیشنهادی DP-ASLA توانست با` — "the proposed DP-ASLA method was able to with".
   The sentence, and the chapter, stop there.
5. **No figures.** The document contains zero drawings, yet the text refers to charts. Either
   the charts must be inserted or the references removed.

---

## 11. F11 / F12 — ε is not implemented, and clipping is not privacy

Tables ۴-۴ and ۴-۵ report an "ε میانگین" column (∞, ۱.۰, ۱.۰۷). The code exposes
`--dp-sigma`, `--dp-clip`, `--dp-delta` — there is **no ε parameter and no accountant**.
The ε values in the document are therefore not computed by anything in this repository.

Related: `clip_only_c2` (clipping with no noise) costs **+0.57 pp** MAPE (3.46 → 4.03), which
is essentially the entire cost attributed to DP (3.46 → 3.96/4.01). In other words, **the
utility cost is dominated by the clipping bound, not by the noise**: σ = 0.001 and σ = 0.01
are statistically indistinguishable (3.96 vs 4.01, sd ≈ 0.07). That is a real and
publishable finding, and it is the opposite of what the document implies. Clipping alone
provides no differential-privacy guarantee and must not be labelled as a privacy mechanism.

---

## 12. What the artifacts *do* support

These claims are safe to make, because each is backed by a committed file:

1. **Convergence.** The no-DP method converges from 99.97 % (round 1) to 3.48 % (round 100)
   on 10 clients, monotonically after round 5.
2. **DP cost is small and clipping-dominated.** 3.46 ± 0.05 % → 3.96 ± 0.08 % (σ=0.001) and
   4.01 ± 0.06 % (σ=0.01): a **+0.50 to +0.55 pp** cost, i.e. ~15 % relative. Clipping alone
   accounts for +0.57 pp, so the noise contributes nothing measurable at these σ.
3. **Seed variance is small.** sd ≤ 0.08 pp across four seeds, well below the DP effect, so
   the effect is not seed noise (n = 4, no formal test — say so).
4. **The attack is infeasible under the stated threat model**, with a quantified diagnosis
   (oracle margin +0.0118 vs. threat-model margin −0.0001; −0.9 % of the ceiling survives).
5. **The replay mechanism is correct**: bit-exact for the noise-free arm (100 % of cells),
   float-rounding-exact for clipping (rel_L2 1.16e-07), and noise-explained for the
   Gaussian arms (residual/noise = 1.0000).

---

## 13. Action list

**Must fix before submission**

1. Regenerate every number in chapter ۴ from a single, logged run; delete or re-derive
   11.29 / 12.24 / 13.60 / 14.20 / 78.52 / 11.5 / 11.2 / 12.0.
2. Run `run_fedavg_only.py` and save its output, or remove the FedAvg row.
3. Recompute table ۴-۱ from one stated window and state the window in the caption.
4. Rewrite the privacy section as a **negative result** with the gate's diagnosis.
5. Restore ۴-۳ / ۴-۳-۱ as `heading 3`, fix ۳-۳ to `heading 3`, finish the truncated
   conclusion, and convert tables ۴-۱…۴-۵ to real Word tables.

**Should fix**

6. Whitelist `*_hourly.csv` in the loader; drop `pjm_hourly_est.csv`, `PJME`, `PJMW`,
   `PJM_Load`; replace `✓` with ASCII.
7. Re-run the four baselines after the seed fix and report the new numbers.
8. Either implement an ε accountant or remove the ε column.
9. Disclose (or fix) the non-aligned client windows.

---

## 14. Files produced by this audit

| file | purpose |
|---|---|
| `_docx_dump.py` | stdlib `.docx` reader (paragraph text + style) |
| `_seed_check.py` | proves Keras 3 initializers ignore `np`/`tf` seeding |
| `_audit_loader.py` | which CSVs the loader accepts as clients |
| `_audit_window.py` | descriptive stats of the real 13,896-sample window |
| `_audit_table41.py` | tests table ۴-۱ against four candidate windows |
| `_audit_windows.py` | per-client calendar windows |
| `_audit_loader.txt`, `_audit_window.txt`, `_audit_table41.txt`, `_audit_windows.txt` | their outputs |

**The `.docx` was not modified.** It is the user's live thesis file and its numbers require
decisions (regenerate vs. remove) that only the author can make.

# AAMOS-00 concordance analysis, v2

Revision of *Characterization of Concordant Users of Smart Inhalers in Asthma* after the JMIR review.
Everything below is reproducible from this repo in about 90 minutes on a laptop; no cloud compute.

**Headline.** Under the revised primary definition, the concordant users are **702 and 917**.
Under the complete-case variant they are **294, 473 and 702**. Under Poisson imputation nobody clears
the threshold, although 294, 473 and 702 stay significant in every imputation.

---

## 1. Pipeline

```
raw CSVs ─► span ─► window join ─► absence case ─► categorise (+ zero filter) ─► correlate ─► summarise ─► permutation test ─► classify ─► sensitivity grid
```

| Stage | What happens | Variants | Assumptions and caveats |
|---|---|---|---|
| **Raw data** | 3 timestamped CSVs, SHA-256 verified on every load | – | Self-report codes are {0, 1, 3, 5, 9, 12}; nothing above 12 |
| **Span** | Per patient, keep only entries inside a date interval | `union` (= v1, trims nothing), `Q`, `D`, `intersection` | A span trims *raw entries*; a patient whose span can't be formed drops out of that world |
| **Window join** | For each questionnaire row, count device records in its window | 11 windows: rolling 12/24/36/48 h, fixed 24 h chunks, calendar day 0/1/2 back | Left join: rows come only from questionnaires. Device records on days with no questionnaire never enter a correlation |
| **Absence case** | What a window with **no device records** means | `A` zero (v1) · `B` Poisson(rate × hours), 10 imputations · `C` drop the window | B also imputes windows with zero self-report (KT8). Zero filter is active **only under A** |
| **Categorise** | Map both columns through one step function | 6 methods; zero filter on/off | Under Spearman 4 methods are rank-identical, so **132 configs = 60 distinct analyses** (30 under B/C) |
| **Correlate** | Spearman and Pearson per config, Fisher Z | both stored | Perfect correlations (\|r\| = 1) are invalid |
| **Summarise** | Per patient, mean or median Z over a config set | `all` (132) or `effective` (60); Spearman or Pearson; mean or median | **Primary = effective / Spearman / mean** |
| **Permutation test** | Shuffle self-report, rerun the pipeline, 10,000 draws or exact enumeration | exact for 328, 398, 454, 917 | Two-tailed **about the null's own centre**; ties count as exceedances |
| **Classify** | Concordant = Z ≥ 0.5 **and** Bonferroni p < 0.05 | thresholds 0.1–0.9 swept | Threshold is on raw observed Z |
| **Grid** | 4 spans × (A + B×10 + C) = 48 worlds × 8 specs × 5 thresholds | – | Set stability = exact set equality, joiners and leavers both count |

### Why the null is not centred on zero

The zero filter drops rows where both signals are zero. After shuffling, a zero self-report that survives is
always paired with a positive device count, which manufactures a negative correlation. Null means reach −0.67
(343), −0.55 (702), −0.50 (190). Case C removes the mutual zeros and the null sits at 0 for everyone, which
confirms the mechanism. v1 compared |null| ≥ |observed| and was therefore sensitive to this shift; the figure
already drew the band around the null's centre, so the code now does the same.

---

## 2. What changed from v1

| Change | Why | Effect on v1 numbers |
|---|---|---|
| Baseline is the `union` span | v1 trimmed nothing | none |
| Centred permutation test | see above; matches the null-band figure | 190, 343, 917 become Bonferroni-significant under the v1 config set |
| Ties count as exceedances | exact-enumeration p must not depend on rounding | ≤ 1/(n+1) per patient |
| Perfect correlations invalid | scipy sometimes returns 1 − 1 ulp; v1 recorded Z ≈ 18 for **7,850 permutations** (398: 4,734; 328: 2,156; 454: 956; 917: 4) | null means for 398/328 were distorted; no significance flips |
| Primary summary over 60 effective configs | 4 of 6 methods are rank-identical under Spearman; the mean over 132 counted them 4× | 294 and 473 fall from 0.52 to 0.46 and 0.43 |
| Exact enumeration for small n | 328 has 136 arrangements, 454 has 840, 917 has 2,520, 398 has 12,650 | no Monte Carlo error for those four |
| Vectorised engine | join is permutation-invariant | 95 s per world locally instead of Cloud Run |

---

## 3. Results

Assessed patients: 113, 190, 294, 328, 343, 398, 447, 454, 473, 514, 625, 701, 702, 917, 939 (15).
Bonferroni α = 0.05/15.

### 3.1 Baseline world (`union`, case A), primary spec

| patient | Z (60 configs) | null mean | distance (null SD) | p Bonf | concordant |
|---|---|---|---|---|---|
| 702 | 0.69 | −0.55 | 10.9 | 0.001 | **yes** |
| 917 | 0.99 | −0.19 | 3.8 | 0.012 | **yes** |
| 294 | 0.46 | −0.01 | 11.4 | 0.001 | no (Z < 0.5) |
| 473 | 0.43 | −0.13 | 7.0 | 0.001 | no (Z < 0.5) |
| 190 | 0.35 | −0.50 | 5.1 | 0.001 | no (Z < 0.5) |
| 343 | −0.06 | −0.67 | 5.6 | 0.001 | no (Z < 0.5) |

Nine of 15 are Bonferroni-significant; the threshold does the selecting.
Median variant: 190, 294, 702, 917. v1 config set (132) with the centred test: 294, 473, 702, 917.

### 3.2 Sensitivity grid, primary spec, threshold 0.5

| span | A | B (always concordant; k/10) | C |
|---|---|---|---|
| union | 702, 917 | – (702: 2/10) | 294, 473, 702 |
| D | 702, 917 | – (702: 4/10) | 294, 473, 702 |
| Q | 702 | – | 294, 473, 702 |
| intersection | 702 | – | 294, 473, 702 |

- **Absence case dominates; span barely matters.** 917 loses concordance only under Q/intersection, where its
  day-0 device records (before its first questionnaire on day 1) are trimmed: Z 0.99 → 0.77, p 0.012 → 0.06.
- **C:** null centred at 0 for everyone; 294 (0.53) and 473 (0.54) clear the threshold; 917 keeps 20 configs and
  is not significant.
- **B:** 294, 473, 702 are significant in **10 of 10** imputations on every span, with means 0.40–0.51.
  B moves the threshold, not the evidence of association. 917 is never significant under B.

### 3.3 Threshold sweep (baseline, primary spec)

| t | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|---|
| concordant | 8 patients | 6 | 702, 917 | 917 | 917 |

### 3.4 Permutation count

Every primary-spec decision in the A and C worlds settles by **2,400** sampled permutations under a
risk-bounded sequential rule (ε = 10⁻³), with resampling risk < 10⁻⁶ at 10,000. A Bonferroni decision cannot
be reached below ~300 draws. Borderline decisions (risk up to 0.44) occur only in B worlds and non-primary specs.

### 3.5 Missingness diagnostics

- **Non-response is not hiding heavy use.** Device use is *lower* on days without a questionnaire for 14 of 15
  patients (significant for 4); non-response comes in blocks for 9 of 15. Median non-response rate 23 %.
- **Device-only days** (never in any correlation): 473 has 40 days / 99 puffs, 917 20 / 64, 939 24 / 70.
- **Mutual zeros** dominate 190 (144 of 163 rows), 343 (146 of 173), 702 (102 of 124).
- **Span D** cuts 190 to 19 rows, 454 to 2, 625 to 16, 701 to 30; unchanged for 294, 473, 917, 939.

---

## 4. Paper updates

| # | Section | Change |
|---|---|---|
| 1 | Methods, multiverse | "132 configurations" → 132 specified, **60 distinct** under Spearman (30 under B/C). Table 2: the ">12 puffs" column never applies; the data-driven methods differ from plain ones only for device counts |
| 2 | Methods, missing data | Missing questionnaires are **excluded**, not treated as zero. Report the non-response check (3.5) |
| 3 | Methods, test | Two-tailed about the null centre; report null mean and distance per patient; explain the shift (§1). Exact enumeration for 4 patients; sequential stopping at 2,400 |
| 4 | Methods, definition | Concordant = mean Z over the 60 effective configs ≥ 0.5 and Bonferroni p < 0.05. State the span (union) and absence case (A) explicitly |
| 5 | Results | Headline: 702 and 917. Grid table (3.2), threshold table (3.3), B as k/10 |
| 6 | Sensitivity | Move into Methods and Results; it is the main analysis, not a post-hoc check |
| 7 | Discussion | Reframe as exploratory; drop causal wording on Clinical Care Value; feedback comparison now rests on n = 2 |
| 8 | Housekeeping | Fill the XX placeholders; hybrid thresholds are per patient, not across patients; prune Pearson outputs; cut or clarify the "Effect on concordance" column |

## 5. Reviewer replies (draft, one line each)

| Reviewer point | Reply |
|---|---|
| R1 small sample | Agreed; conclusions reframed as exploratory; headline set is now 2 (or 3 under C) of 15 |
| R2 too many analyses | Multiverse reduced to its 60 distinct configurations; clustering and embedding classifier to be cut; grid reported as two marginals |
| R3 arbitrary threshold | Threshold swept 0.1–0.9 (3.3); mean/median both reported; classification stability shown per world |
| R4 absence = zero is strong | Three absence cases run on four spans (3.2); non-response mechanism examined directly (3.5) |
| R5 causality | Wording removed |
| R6 definition ambiguous | Definition now one sentence (§4 row 4) with every choice named |
| R7 sensitivity is post hoc | Moved into Methods; it is the analysis |

Kevin's comments: KT1/KT2 answered by 3.4; KT3 by the B/C filter coupling; KT5 by §4 row 2; KT8 implemented;
KT11/KT13 by the span axis; KT14 by the cell-count tables; KT16 by exact set equality in the grid.

---

## 6. Reproduce

```bash
python -m aamos_concordance --verify                       # raw data matches the manifest
python scripts/build_world.py --world span=union__case=A   # one world, ~95 s
python scripts/sensitivity_grid.py                          # grid + marginals over built worlds
python scripts/missingness_diagnostics.py                   # 3.5
python run_all.py --changed-worlds --skip-sentiment         # downstream comparisons where the set changed
python -m pytest -q                                         # 601 tests, ~10 min
```

## 7. Where things are

| Path | Contents |
|---|---|
| `aamos_concordance/` | the pipeline: `join`, `categorization`, `engine`, `spans`, `summary`, `sensitivity`, `resampling` |
| `results/v1/` | frozen published results |
| `results/v2/worlds/<span>__<case>/` | per world: `per_config_z.csv`, `summary.csv`, `observed.csv`, `threshold_sweep.csv` |
| `results/v2/sensitivity/` | `grid.csv`, `world_stability.csv`, `case_b_stability.csv`, `threshold_curve.csv` |
| `results/v2/diagnostics/` | cell counts, non-response check, effective configurations |
| `results/v2/groups/concordant_sets.json` | every set, per world and spec; `config.py` reads groups from here |
| `tests/oracles/` | verbatim v1 code; every refactor is pinned to it |

Every output has a `.provenance.json` sidecar with the git commit, data hashes and configuration.

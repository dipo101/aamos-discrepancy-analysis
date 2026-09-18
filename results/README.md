# Results layout

```
results/
  v1/        Frozen results of the published (pre-v2) analysis. Not regenerated;
             the only scripts that write here are the v1 figure scripts
             (permutation_cloudrun/aggregator/create_*_plots.py,
             run_sensitivity_analysis.py, per_patient_analysis_by_measure.py),
             which redraw figures from the frozen tables.
  v2/
    worlds.csv                    Index of every world that has been built:
                                  parameters, build time, git commit, data hashes.
    worlds/<world_id>/            One folder per world, named by its parameters,
                                  e.g. span=union__case=A (the baseline = v1: the union span trims
                                  nothing; Q/D/intersection trim raw entries outside the period).
      config.json                 The world spec and how it was built.
      per_config_z.csv            Observed multiverse: patient x 132 configs x Fisher Z.
                                  Written by scripts/build_world.py --world ... (engine)
      null.parquet                Null distribution: patient x permutation x summary stats.
                                  Written by scripts/build_world.py (engine, local; git-ignored)
      summary.csv                 Long table: patient, measure, observed Z, permutation
                                  p-value, Bonferroni, n_ties_at_observed. Derived from the two above.
    groups/concordant_sets.json   {world_id: {mean: [...], median: [...]}}. GENERATED from
                                  summary.csv; config.py reads groups from here. Never hand-edit.
    comparisons/<world_id>/<group>/<analysis>/
                                  Downstream outputs (demographics, temporal, Bland-Altman,
                                  feedback). Each folder has RUN.provenance.json.
    diagnostics/                  Cell counts and missingness checks (v2 items 5-6).
```

Every CSV/parquet has a `<name>.provenance.json` sidecar recording the git commit,
the raw-data hashes and the configuration that produced it.

The world id is the primary key. `python -c "from aamos_concordance.worlds import list_worlds; print(list_worlds())"`
lists what exists; `pandas.read_csv('results/v2/worlds.csv')` gives the index.

`tests/test_baseline_world.py` checks that the baseline world reproduces v1:
its per-config Z table is v1's verbatim, its summary rebuilds from its inputs,
and the derived sets are the ones the manuscript reports.

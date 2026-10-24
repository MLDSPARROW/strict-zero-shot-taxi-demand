# Verification: where every table and figure number comes from

This file maps every numbered table and figure in the paper to the exact script that produced it, the
exact output file in this repository that contains the matching numbers, and the commit that introduced
that output file. It exists so that a reader can check a number without rerunning anything, and can rerun
the exact script if they want to reproduce it themselves. Commit hashes are short SHAs on `main`
(`git show <hash>` or `git log --oneline` to look one up).

| Paper item | Script | Result file in this repo | Introduced in commit |
|---|---|---|---|
| Fig. 1, pipeline overview | — (hand-drawn TikZ diagram, not derived from data) | — | — |
| Table "Re-aggregation of the share helpers" (`tab:k`) | `improve/aggregate.py`, `improve/aggregate_multi.py`, `improve/aggregate_multi2.py` | printed to console at build time; **not currently archived as a results file** — rerun these scripts to reproduce | 8df1116 (scripts) |
| Fig. "Scale-matched re-aggregation" (`fig:scale`) | `improve/figures.py` | `figs/fig_scale.pdf` (image, not a text log) | 8df1116 (script), features in `features/` |
| Table 1, data (`tab:data`) | `improve/paper_numbers.py` | `results/paper_numbers.txt`, section "TABLE 1: data" | 8df1116 |
| Table 2, main development results (`tab:main`) | `improve/predict_final.py` + `improve/evaluate_strict.py` | `results/eval_dev.txt` | ddc3d54 |
| Fig. "Component bars" (`fig:steps`) | `improve/scale_test.py`, `improve/figures.py` | `results/scale_test.txt` | ddc3d54 |
| Fig. "Observed vs predicted maps" (`fig:maps`) | `improve/figures.py` | `figs/fig_maps.pdf`, built directly from `improve/out_cal/pred_<city>.npz` (image, not a text log) | 8df1116 |
| Fig. "Week of demand" (`fig:week`) | `improve/figures.py` | `figs/fig_week.pdf`, same source as above | 8df1116 |
| Table "Allocation-only metrics" (`tab:alloc`) | `improve/diagnostics_cal.py` | `results/diag_dev.txt` (Chicago, NYC), `results/diag_sf.txt`, `results/diag_dc.txt` | ddc3d54 |
| Table "Block-size sweep" (`tab:sweep`) | `improve/aggregate_sweep.py`, `improve/sweep_eval.py` | `results/aggregate_sweep.txt`, `results/sweep_eval.txt` | ddc3d54 |
| Fig. "Sweep curve" (`fig:sweep`) | `improve/sweep_eval.py` | `figs/fig_sweep.pdf`; numbers in `results/sweep_eval.txt` | ddc3d54 |
| Table "Aggregation control" (`tab:control`) | `improve/aggregate_control.py`, `improve/control_eval.py` | `results/aggregate_control.txt`, `results/control_eval.txt` | ddc3d54 |
| Section "Scale matching with a non-neural allocator" (XGBoost check) | `improve/xgb_scale.py` | `results/xgb_scale.txt` | 9f1f63e |
| Table "Held-out targets" (`tab:heldout`) | `improve/predict_final.py` + `improve/evaluate_strict.py` | `results/eval_sf.txt`, `results/eval_dc.txt` | ddc3d54 |
| Fig. "Four-city comparison" (`fig:four`) | `improve/fig_four.py` | `figs/fig_four.pdf`; underlying numbers in `results/eval_dev.txt`, `results/eval_sf.txt`, `results/eval_dc.txt` | ddc3d54 |
| Table "Level predictor selection" (`tab:level`) | `improve/v4_numbers.py` (section "(1) LEVEL unit selection"); also printed by `improve/predict_final.py` | `results/v4_numbers.txt` | 7cf58fa |
| Table "Sources of error" (`tab:headroom`) | `improve/diagnostics_cal.py`, section "(1)" | `results/diag_dev.txt`, `results/diag_sf.txt`, `results/diag_dc.txt` | ddc3d54 |
| Table "Contextual comparison with published results" (`tab:lit`) | This work's own rows: `improve/evaluate_strict.py`, `improve/cost_final.py` | `results/eval_dev.txt`, `results/cost_final.txt`. The other rows are copied from the cited papers' own tables (OpenCity, UrbanGPT, GPT-ST) and are not reproducible from this repository. | ddc3d54, 8df1116 |
| Table "Cost of strictness" (`tab:cost`) | `improve/cost_final.py` | `results/cost_final.txt` | 8df1116 |
| Fig. "Cost of strictness" (`fig:cost`) | `improve/cost_final.py`, `improve/figures.py` | `figs/fig_cost.pdf`; numbers in `results/cost_final.txt` | 8df1116 |

Items added on 21 October 2026 (analyses after the main evaluation; each was run under a time-stamped protocol,
except the two descriptive items marked as such in the paper):

| Paper item | Script | Result file in this repo | Introduced in commit |
|---|---|---|---|
| Table "Additional targets" (`tab:data2`) | `improve/build_coarse_targets.py` (coarse versions), `multicity/prep_austin.py` (Austin cells) | region counts and totals are printed in `results/coarse_eval.txt` and `results/austin_eval.txt` | 7e3eaba, d094124 |
| Table "Coarse versions of the held-out cities" (`tab:coarse`) | `improve/build_coarse_targets.py`, `improve/runner_coarse.py`, `improve/coarse_eval.py`; Chi row: `improve/chi_baseline.py`, `improve/chi_eval.py` | `results/coarse_eval.txt`, `results/chi_eval.txt`; protocol `improve/PROTOCOL_COARSE_TARGETS.md` | 7e3eaba, dc206da, 9851b03 |
| Table "Austin" (`tab:austin`) | `multicity/prep_austin.py`, `improve/build_extra.py`, `improve/runner_austin.py`, `improve/austin_eval.py`; forecast-of-zero row: `improve/v1_i1_eval.py` | `results/austin_eval.txt`, `results/v1_i1_eval.txt`; protocol `improve/PROTOCOL_MORE_CITIES.md` | d094124, f82de81, 8cb5e45 |
| Table "Gain of scale matching in all targets" (`tab:scalegain`) and Fig. (`fig:scalegain`) | `improve/scale_gain_plot.py`, which parses `results/control_eval.txt`, `results/coarse_eval.txt` and `results/austin_eval.txt` (descriptive, computed after the individual results) | `results/scale_gain.txt`; `figs/fig_scalegain.pdf` | 47c7479 |
| Table "Block-size sweep for Washington, DC" (`tab:sweepdc`) and the sentences on it in the protocol, control and discussion sections | `improve/build_scale_rules.py`, `improve/runner_scale_rules.py`, `improve/scale_rule_eval.py`, `improve/dc_sweep_table.py` | `results/scale_rule_eval.txt`, `results/dc_sweep_table.txt`, `results/scale_rules_build.txt`; protocol `improve/PROTOCOL_V3.md` | see the commit that adds this row |
| Section "Comparison with the allocation of Chi et al." (numbers in the text) | `improve/chi_baseline.py`, `improve/chi_eval.py` | `results/chi_eval.txt`; protocol `improve/PROTOCOL_CHI_BASELINE.md` | dc206da, 7babd5c |
| `tab:lit`: hourly rows of this work for Chicago, NYC, Washington, DC and Austin; forecast-of-zero values in the limitations | `improve/hourly_compare.py`, `improve/v1_i1_eval.py` (descriptive, protocol `PROTOCOL_V2.md` addendum 5) | `results/hourly_compare.txt`, `results/v1_i1_eval.txt` | cb6a760, f82de81 |
| Austin rides per transit commuter (113) and the taxi-city range (15.1 to 19.5) in the Austin section | `improve/explore_scale_stats.py` (Table 1 gives the taxi cities; Austin: 900,643 rides in 248 days scaled to a year, ACS transit commuters) | `results/explore_scale_stats.txt` | 21fe0f2 |
| Table "Final method and XGBoost allocator" (`tab:xgbheld`) | `improve/xgb_heldout.py` (ten XGBoost runs per held-out target on the scale-matched helpers), scored by `improve/v10_heldout.py` | `results/v10_heldout.txt`, rows "XGBoost, scale-matched" and "final method v1"; protocol `improve/PROTOCOL_V10.md` (written before any XGBoost model was trained for these targets) | 24 Oct 2026 |
| Fig. "four targets" (`fig:four`), bars of the XGBoost allocator | `improve/fig_four.py` (values typed in from the files named in its docstring: `results/xgb_scale.txt` for Chicago and NYC, `results/v10_heldout.txt` for SF and DC) | `figs/fig_four.pdf` | 24 Oct 2026 |
| Version D of `tab:cost` and `fig:cost` (Chronos-Bolt, previous 7 days) | `improve/fm_chronos.py` (package chronos-forecasting in the separate environment `.venv_fm`), figure `improve/fig_cost.py` | `results/fm_chronos.txt`; protocol `improve/PROTOCOL_V6.md` (written before any forecast) | 22 Oct 2026 |
| Table "Masked demand modelling" (`tab:masked`) | `improve/run_x.py` (switch `d`), `improve/v4_eval.py` (confirmation), `improve/v4_dev_strict.py` (strict scoring), `improve/runner_v4_heldout.py` (held-out runs) | `results/v4_dev_strict.txt`, `results/v4_confirm.txt`, `results/v4_heldout.txt`; protocol `improve/PROTOCOL_V4.md`, addenda 1 to 3 | 22 to 24 Oct 2026 |

Rows of `tab:lit` that come from other studies are copied from the cited tables (Chen et al. 2023 Tab. 2,
Bansal et al. 2021 Tab. 3, Hou et al. 2019 Tab. 1, in addition to those listed above) and are not reproduced here.

## Known gap

None at present. `tab:k` (the re-aggregation sizes, base partition) was originally only reproducible by
rerunning `improve/aggregate.py`, because its console output had not been saved. After the data pipeline was
rebuilt on 20 October 2026, that output was archived as `results/aggregate.txt` (block counts and median block
areas: NYC 263 to 101 and SF 243 to 21 blocks for Chicago, SF 243 to 60 blocks for NYC), and the rebuilt
files reproduce the paper's block counts exactly. The ten partitions of every helper are rebuilt by
`aggregate_multi.py` and `aggregate_multi2.py`; their block counts and median areas are printed there.

The local working data of the earlier work (not the repository) were lost between 3 and 17 October 2026 and
were rebuilt from public sources; the rebuilt trip totals match the paper's Table 1 exactly (checked in
`improve/build_coarse_targets.py`), and the rebuilt final-method predictions reproduce the paper's results
within 0.005 MAE (`results/v1_i1_eval.txt`, "reproduction check" lines).

## How to check a number yourself without rerunning anything

1. Find the paper item in the table above.
2. Open the listed file in `results/`.
3. The script's printed output is plain text; every number quoted in the paper appears there, typically to
   two decimal places as in the paper.
4. To confirm the file has not changed since it was committed: `git log --oneline -- results/<file>` shows
   every commit that touched it; `git show <commit>:results/<file>` shows its content at that commit.

## How to reproduce a number from scratch

Follow `README.md`'s pipeline table to rebuild the public data and features, then run the script named
above. Every script that reads target-city demand is named explicitly in the "Reads target demand?" table
in `README.md`; everything else only ever reads helper-city data and public covariates.

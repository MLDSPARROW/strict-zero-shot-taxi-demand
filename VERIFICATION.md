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

## Known gap

`tab:k` (the re-aggregation sizes, Chicago partition 1) is currently only reproducible by rerunning
`improve/aggregate.py`; its console output was not saved as a `results/*.txt` file at the time it was run.
This is flagged here rather than silently left out, and will be fixed the next time the data pipeline is
rebuilt (see `PROTOCOL_REVISION.md` and the "more cities" follow-up work), by saving that script's output
to `results/aggregate.txt`.

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

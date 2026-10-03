# Non-neural scale-matching check (written 3 October 2026, before any run described here)

Question: does scale matching also help a model that is not the neural allocator?

Model: the XGBoost baseline of the paper, unchanged (baselines.py / paper_numbers.py): 18 map features and 9 calendar
features, features scaled by median and IQR pooled over the helper regions, target = region share times the number of
regions, Tweedie loss, 400 trees, depth 6, learning rate 0.05, subsample 0.8, colsample 0.8, 800 random slots per helper,
predictions normalised per slot.

Targets: Chicago (helpers New York City, San Francisco) and New York City (helpers Chicago, San Francisco).

Conditions, 10 runs each, averaged:
- native: native helpers, run r uses XGBoost seed r-1 and slot-sampling seed r-1;
- scale-matched: helpers merged exactly as in the final method, run r uses partition r (Chicago stays native as a
  helper of New York City), XGBoost seed r-1 and slot-sampling seed r-1.

Scoring: fully strict, full year, trip MAE and RMSE with the strict city totals of predict_final.py, and the mean daily
MAE difference (scale-matched minus native) with a 7-day block-bootstrap 95% interval (2,000 draws).
Both outcomes are reported, whatever they are.

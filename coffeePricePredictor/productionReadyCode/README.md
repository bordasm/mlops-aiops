# Coffee quality benchmarking pipeline

This folder contains a production-ready Python implementation of the workflow developed in the notebook `coffee-quality-pricing-benchmark.ipynb`.

## Project structure

- `main.py` trains the benchmark model on the provided dataset and saves artifacts.
- `src/coffee_price_predictor/` contains the reusable pipeline modules.
- `artifacts/` stores the trained model, preprocessing pipeline, and evaluation metrics.

## Run training

From this directory:

```bash
python main.py
```

Optional arguments:

```bash
python main.py --data-path "C:/path/to/coffee_value_benchmark.csv" --output-dir "artifacts"
```

## Output artifacts

- `coffee_quality_model.joblib` – selected best regressor model
- `preprocessor.joblib` – preprocessing transformer for numeric and categorical columns
- `model_summary.json` – model comparison with RMSE, MAE, and R-squared metrics

## Notes

- The modeling workflow drops known leakage columns (`record_id`, `lot_id`, `quality_grade`, `price_usd_per_kg`).
- Missing numeric values are imputed using medians and categorical values are imputed using mode.
- The final selected model is the highest-performing regressor based on validation R-squared.

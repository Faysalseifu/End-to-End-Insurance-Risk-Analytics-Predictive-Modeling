# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling

Analyze South African car insurance data to identify low-risk segments. Perform EDA, A/B hypothesis testing (province, postal code, gender), build claim severity & premium prediction models using XGBoost, and implement DVC + Git workflows. Deliver actionable insights to optimize pricing and marketing.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## End-to-end workflow (Task 1)

1) **Run EDA artifacts (scriptable)**

```bash
python -m scripts.eda --data data/insurance.csv --out reports/eda
```

Outputs: `summary_stats.csv`, `missing_values.csv`, `correlation.csv`, and PNG plots in `reports/eda/`. The helper uses the same validation as the notebook and will raise clear errors if the CSV is missing or empty.

2) **Prepare features + train models (interactive snippet)**

```python
from scripts.data_processing import load_and_clean_data, encoder, scaler
from scripts.model import split_data, train_models, evaluate_model

df = load_and_clean_data("data/insurance.csv")
df = encoder("oneHotEncoder", df, columns_label=[], columns_onehot=["sex", "smoker", "region"])
df = scaler("standardScaler", df, columns_scaler=["age", "bmi"])

X = df.drop(columns=["charges"])  # adjust target as needed
y = df["charges"]
X_train, X_test, y_train, y_test = split_data(X, y)
lr, dt, rf, xgbm = train_models(X_train, y_train)
metrics = evaluate_model(lr, X_test, y_test)
```

All helpers now perform minimal input checks (e.g., missing columns, empty data) and raise friendly errors instead of failing silently.

## Running EDA notebook

- Open `notebooks/Exploratory Data Analysis.ipynb` in Jupyter Lab or VS Code.
- Run cells sequentially; plots cover distributions, outliers, smoker impact on charges, and feature correlations.

## CI/CD

- GitHub Actions workflow (`.github/workflows/ci.yml`) installs dependencies, compiles Python files, and runs pytest (tolerant if no tests are present). Triggers on pushes/PRs to `main` and `task-1`.

## Data notes

- Current sample dataset (`data/insurance.csv`) includes demographics, BMI, smoker flag, region, and charges. It lacks premium, claim frequency, vehicle, geography granularity, and dates—so loss-ratio and temporal trend analyses from the brief cannot be computed until richer data is added.

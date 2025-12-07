# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling

Analyze South African car insurance data to identify low-risk segments. Perform EDA, A/B hypothesis testing (province, postal code, gender), build claim severity & premium prediction models using XGBoost, and implement DVC + Git workflows. Deliver actionable insights to optimize pricing and marketing.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running EDA notebook

- Open `notebooks/Exploratory Data Analysis.ipynb` in Jupyter Lab or VS Code.
- Run cells sequentially; plots cover distributions, outliers, smoker impact on charges, and feature correlations.

## CI/CD

- GitHub Actions workflow (`.github/workflows/ci.yml`) installs dependencies, compiles Python files, and runs pytest (tolerant if no tests are present). Triggers on pushes/PRs to `main` and `task-1`.

## Data notes

- Current sample dataset (`data/insurance.csv`) includes demographics, BMI, smoker flag, region, and charges. It lacks premium, claim frequency, vehicle, geography granularity, and dates—so loss-ratio and temporal trend analyses from the brief cannot be computed until richer data is added.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.inspection import permutation_importance




# 1. Load the CSV, parsing the date column and using it as the index
df = pd.read_csv('weather_data.csv', parse_dates=['date'], index_col='date')

# 2. Sanity checks
print(df.index.min(), "->", df.index.max())   # should be 2015-01-01 -> 2025-12-31
print(df.shape)                               # about 4,000 rows
print(df.isna().sum())                        # missing values per column
print(df.describe())                          # min/max/mean look plausible?

# 4. Plot the full 10-year series
plt.figure(figsize=(10,7))
ax = df["temperature_2m_max"].plot(linewidth=0.8)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Maximum Temperature (2m) - 10-Year Series")
plt.show()

# 5. Build the target: tomorrow's max temperature
df['target'] = df['temperature_2m_max'].shift(-1)
# 6. The persistence prediction: tomorrow = today
df['pred_persistence'] = df['temperature_2m_max']
# 7. The last day has no tomorrow, so drop it
df = df.dropna()
# 8. Eyeball the alignment before trusting any number
print(df[['temperature_2m_max', 'target', 'pred_persistence']].head())
# 9. Compute the MAE
mae_manual = (df['target'] - df['pred_persistence']).abs().mean()
mae_sklearn = mean_absolute_error(df['target'], df['pred_persistence'])

print("MAE (manual):", mae_manual)
print("MAE (sklearn):", mae_sklearn)

# 10. Lag features (t-1, t-2, t-3)
df["t_1"] = df["temperature_2m_max"].shift(1)
df["t_2"] = df["temperature_2m_max"].shift(2)
df["t_3"] = df["temperature_2m_max"].shift(3)

# 11. The 7-day rolling mean
df['rolling_mean_7'] = df['temperature_2m_max'].rolling(window=7).mean()

#12. Day-of-year as sin and cos
df['day_of_year'] = df.index.dayofyear
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Drop rows with NaN (the first 6 days and the last day)
features = ['t_1', 't_2', 't_3', 'rolling_mean_7', 'day_of_year_sin', 'day_of_year_cos']
df = df.dropna(subset=features + ['target'])

print(df[features + ['target']].head())
print(df.shape)

# 13. Split into train and test sets
train = df.loc[:"2023"] # from the beginning of the dataset up to 2023(first 8 years)
test = df.loc["2024":] # from 2024 to the end of the dataset(last two years)

X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

print(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} rows)")
print(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} rows)")

# 14. Fit the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 15. Predict and compare on the SAME test days
y_pred = model.predict(X_test)

mae_model = mean_absolute_error(y_test, y_pred)
mae_persistence = mean_absolute_error(y_test, test['pred_persistence'])
improvement = (mae_persistence - mae_model) / mae_persistence * 100


print(f"MAE (model): {mae_model:.2f}")
print(f"MAE (persistence): {mae_persistence:.2f}")
print(f"Improvement over persistence: {improvement:.2f}%")

# 16. Switch to walk-forward validation
X = df[features]
y = df["target"]
# 2. The models to compare (default settings, same seed)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

tscv = TimeSeriesSplit(n_splits=5, test_size=365)

rows = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Baseline on the same test rows
    rows.append({
        "fold": fold,
        "method": "Persistence baseline",
        "mae": mean_absolute_error(y_test, df["pred_persistence"].iloc[test_idx]),
    })

    for name, model in models.items():
        m = clone(model)                      # fresh, unfitted copy every fold
        m.fit(X_train, y_train)
        rows.append({
            "fold": fold,
            "method": name,
            "mae": mean_absolute_error(y_test, m.predict(X_test)),
        })

results = pd.DataFrame(rows)

# 4. Per-fold table and summary
per_fold = results.pivot(index="fold", columns="method", values="mae")
per_fold = per_fold[["Persistence baseline", *models.keys()]]
print("MAE per fold (°C):")
print(per_fold.round(2))

order = per_fold.columns
summary = pd.DataFrame({
    "mean_mae": per_fold.mean(),
    "std_mae": per_fold.std(),
})
baseline_mean = summary.loc["Persistence baseline", "mean_mae"]
summary["improvement_%"] = (baseline_mean - summary["mean_mae"]) / baseline_mean * 100
summary["folds_beating_baseline"] = [
    (per_fold[c] < per_fold["Persistence baseline"]).sum() for c in order
]
print("\nSummary:")
print(summary.round(2))

# Bar chart: mean MAE with std across folds as error bars
colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(order, summary["mean_mae"], yerr=summary["std_mae"],
              capsize=6, color=colors)
ax.bar_label(bars, labels=[f"{v:.2f}" for v in summary["mean_mae"]],
             padding=3, fontsize=10)
ax.set_ylabel("Mean absolute error (°C)")
ax.set_title("Next-day max temperature: walk-forward MAE (5 folds)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("mae_comparison.png", dpi=150)
plt.show()

# 17. Add feature importance
train = df.loc[:"2023"]
test = df.loc["2024":]
X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

# 18. Fit the model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
# 19. Built-in importance (measured on the training data)

builtin = pd.Series(model.feature_importances_, index=features)
# 20. Permutation importance (measured on the unseen test data)
perm = permutation_importance(
    model, X_test, y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)
perm_mean = pd.Series(perm.importances_mean, index=features)   # MAE increase in °C
perm_std = pd.Series(perm.importances_std, index=features)
summary = pd.DataFrame({
    "builtin_importance": builtin,
    "permutation_mae_increase_C": perm_mean,
    "permutation_std": perm_std,
}).sort_values("permutation_mae_increase_C", ascending=False)
print(summary.round(3))

# 21. Side-by-side horizontal bar charts
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

b = builtin.sort_values()
axes[0].barh(b.index, b.values, color="#4c78a8")
axes[0].set_title("Built-in importance\n(share of error reduction in training)")
axes[0].set_xlabel("Importance")

p = perm_mean.sort_values()
axes[1].barh(p.index, p.values, xerr=perm_std[p.index], color="#f58518", capsize=4)
axes[1].set_title("Permutation importance on test set\n(MAE increase when shuffled)")
axes[1].set_xlabel("Increase in MAE (°C)")

for ax in axes:
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# 22. Time-based split (same as step 5)
train = df.loc[:"2023"]
test = df.loc["2024":]
X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

# 23. Method A: quantile regression (10th and 90th percentiles = 80% interval)
lower_model = GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=42)
upper_model = GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=42)
lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

q_lower = lower_model.predict(X_test)
q_upper = upper_model.predict(X_test)

# Guard against quantile crossing (lower > upper)
n_crossed = ( q_lower > q_upper ).sum()
q_lower, q_upper = np.minimum(q_lower, q_upper), np.maximum(q_lower, q_upper)
print(f"Quantile crossing: {n_crossed} days (out of {len(q_lower)})")

# 24. Method B: residual-based interval
# Fit on 2015-2022, compute residuals on 2023 (unseen by the model), apply to the test set
fit_part = df.loc[:"2022"]
calib_part = df.loc["2023"]

point_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
point_model.fit(fit_part[features], fit_part["target"])


residuals = calib_part["target"] - point_model.predict(calib_part[features])
lo_off, hi_off = np.percentile(residuals, [10, 90])

point_pred = point_model.predict(X_test)
r_lower = point_pred + lo_off
r_upper = point_pred + hi_off

def evaluate(y_true, lower, upper, name):
    inside = (y_true.values >= lower) & (y_true.values <= upper)
    coverage = inside.mean()
    width = (upper - lower).mean()
    print(f"{name:<22} coverage: {coverage:.1%}   mean width: {width:.2f} °C")
    return inside

inside_q = evaluate(y_test, q_lower, q_upper, "Quantile regression")
inside_r = evaluate(y_test, r_lower, r_upper, "Residual-based")

# 25. Plot 60 days of the test set with the quantile interval
window = slice(0, 60)
dates = y_test.index[window]

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(dates, q_lower[window], q_upper[window],
                alpha=0.3, color="#4c78a8", label="80% prediction interval")
ax.plot(dates, y_test.values[window], color="black", linewidth=1.5, label="Actual")
ax.scatter(dates[~inside_q[window]], y_test.values[window][~inside_q[window]],
           color="red", zorder=3, s=25, label="Outside interval")
ax.set_ylabel("Next-day max temperature (°C)")
ax.set_title("80% prediction interval (quantile gradient boosting)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("prediction_interval.png", dpi=150)
plt.show()

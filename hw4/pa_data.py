import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# 1) Helpers to make columns look "HRIS-real"
# ---------------------------------------------------------------------
def _minmax_from_percentiles(z: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    """Robust min-max using percentiles (reduces extreme outliers)."""
    a = np.percentile(z, lo)
    b = np.percentile(z, hi)
    zc = np.clip(z, a, b)
    if b - a < 1e-12:
        return np.zeros_like(zc)
    return (zc - a) / (b - a)


def bounded_rate_from_z(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map real-valued z to a bounded [lo, hi] using a sigmoid."""
    s = 1.0 / (1.0 + np.exp(-z))
    return lo + (hi - lo) * s


def positive_skew_from_z(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map z to a positive-skew variable in [lo, hi] (via exp then robust minmax)."""
    e = np.exp(z)
    u = _minmax_from_percentiles(e, 1, 99)
    return lo + (hi - lo) * u


def integer_bucket_from_z(z: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Discretize to integers in [lo, hi] by robust minmax then rounding."""
    u = _minmax_from_percentiles(z, 1, 99)
    vals = lo + (hi - lo) * u
    return np.rint(vals).astype(int)


def zscore(v: np.ndarray) -> np.ndarray:
    """Safe z-score for numeric arrays."""
    v = v.astype(float)
    mu = v.mean()
    sd = v.std()
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


# ---------------------------------------------------------------------
# 2) Generate synthetic classification data (your spec)
#    With shuffle=False, columns are ordered:
#      [0..n_informative-1] informative,
#      [n_informative..n_informative+n_redundant-1] redundant (linear combos),
#      remaining = noise
# ---------------------------------------------------------------------
X, y = make_classification(
    n_samples=130_000,
    n_features=15,
    n_informative=5,
    n_redundant=2,
    n_classes=5,
    n_clusters_per_class=3,
    weights=[0.05, 0.13, 0.6, 0.17, 0.05],
    flip_y=0.08,
    class_sep=0.7,
    hypercube=True,
    shuffle=False,
    random_state=SEED,
)

# Convert target to a 5-point appraisal scale: {1,2,3,4,5}
y = y.astype(int) + 1

# ---------------------------------------------------------------------
# 3) Assign semantic feature names
#    5 informative: plausible predictors of appraisal (KEEP SAME)
#    2 redundant: real-world derived redundancies from informative
#    8 noise: plausible HRIS fields with weak/none direct relationship
#    + NEW: job_satisfaction (1-100) predicted linearly by some noise features
# ---------------------------------------------------------------------
Z = X.copy()

# ---- Informative (Z[:,0..4]) KEEP SAME ----
tenure_years = positive_skew_from_z(Z[:, 0], 0.0, 30.0)  # 0–30 yrs (skewed)
job_level = integer_bucket_from_z(Z[:, 1], 1, 8)  # 1–8
goal_completion_rate = bounded_rate_from_z(Z[:, 2], 0.0, 1.0)  # 0–1
engagement_score = bounded_rate_from_z(Z[:, 3], 0.0, 100.0)  # 0–100
absences_days_12mo = positive_skew_from_z(
    -Z[:, 4], 0.0, 30.0
)  # 0–30 (invert so higher Z -> fewer absences)

# ---- Redundant (overwrite to be "HRIS-real" redundancies of informative) ----
tenure_months = np.clip(
    np.rint(tenure_years * 12 + rng.normal(0, 2.0, size=tenure_years.shape)), 0, 360
).astype(int)
engagement_percent = np.clip(
    engagement_score / 100.0 + rng.normal(0, 0.01, size=engagement_score.shape), 0, 1
)

# ---- Noise (Z[:,7..14]) ----
age_years = np.clip(
    np.rint(18 + (65 - 18) * _minmax_from_percentiles(Z[:, 7], 1, 99)), 18, 65
).astype(int)
commute_minutes = np.clip(
    np.rint(5 + (120 - 5) * _minmax_from_percentiles(Z[:, 8], 1, 99)), 5, 120
).astype(int)
training_hours_12mo = np.clip(
    np.rint(
        0
        + (120 - 0)
        * _minmax_from_percentiles(positive_skew_from_z(Z[:, 9], 0, 1), 1, 99)
    ),
    0,
    120,
).astype(int)
overtime_hours_month = np.clip(
    np.rint(
        0
        + (60 - 0)
        * _minmax_from_percentiles(positive_skew_from_z(Z[:, 10], 0, 1), 1, 99)
    ),
    0,
    60,
).astype(int)
remote_work_ratio = bounded_rate_from_z(Z[:, 11], 0.0, 1.0)  # 0–1
team_size = np.clip(
    np.rint(2 + (25 - 2) * _minmax_from_percentiles(Z[:, 12], 1, 99)), 2, 25
).astype(int)
department_code = integer_bucket_from_z(Z[:, 13], 1, 12)  # 1–12 (code)
last_promotion_years_ago = np.clip(
    np.rint(
        0
        + (15 - 0)
        * _minmax_from_percentiles(positive_skew_from_z(Z[:, 14], 0, 1), 1, 99)
    ),
    0,
    15,
).astype(int)

# ---------------------------------------------------------------------
# 4) NEW FEATURE: job_satisfaction (1–100), linearly predicted by some noise features
#    Requirements:
#      - negative correlation with commute_minutes
#      - positive with training_hours_12mo
#      - positive with remote_work_ratio
#      - negative with last_promotion_years_ago
#
#    We'll create a latent linear score in z-space from standardized versions of these predictors,
#    add a bit of noise, then map through sigmoid to 1–100.
# ---------------------------------------------------------------------
# Standardize predictors
z_commute = zscore(commute_minutes)
z_training = zscore(training_hours_12mo)
z_remote = zscore(remote_work_ratio)
z_promo = zscore(last_promotion_years_ago)

# Coefficients chosen to enforce directionality and moderate strength
# (You can tweak these if you want stronger/weaker relationships)
b_commute = -0.90
b_training = +0.60
b_remote = +0.70
b_promo = -0.55

# Latent linear satisfaction score + noise
latent_sat = (
    b_commute * z_commute
    + b_training * z_training
    + b_remote * z_remote
    + b_promo * z_promo
    + rng.normal(0, 0.50, size=z_commute.shape)  # noise term
)

# Map to 1–100 similar to engagement_score
job_satisfaction = bounded_rate_from_z(latent_sat, 1.0, 100.0)

# ---------------------------------------------------------------------
# 5) Build final DataFrame
# ---------------------------------------------------------------------
df = pd.DataFrame(
    {
        # Informative (5)
        "department_code": department_code,
        "age_years": age_years,
        "tenure_months": tenure_months,
        "tenure_years": tenure_years,
        "overtime_hours_month": overtime_hours_month,
        "remote_work_ratio": remote_work_ratio,
        "job_level": job_level,
        "goal_completion_rate": goal_completion_rate,
        "engagement_score": engagement_score,
        # New derived feature (from noise predictors)
        "job_satisfaction": job_satisfaction,
        "absences_days_12mo": absences_days_12mo,
        "engagement_percent": engagement_percent,
        "commute_minutes": commute_minutes,
        "training_hours_12mo": training_hours_12mo,
        "team_size": team_size,
        "last_promotion_years_ago": last_promotion_years_ago,
        # Target
        "performance_rating": y,
    }
)

# Round floats to look like typical exported HRIS extracts
float_cols = [
    "tenure_years",
    "goal_completion_rate",
    "engagement_score",
    "absences_days_12mo",
    "engagement_percent",
    "remote_work_ratio",
    "job_satisfaction",
]
df[float_cols] = df[float_cols].round(3)

# ---------------------------------------------------------------------
# 6) Train / hidden-test split (stratified by target) + save CSV
# ---------------------------------------------------------------------
train_df, hidden_test_df = train_test_split(
    df, test_size=0.25, stratify=df["performance_rating"], random_state=SEED
)

# Sanity checks: class proportions
print(
    "Overall:",
    df["performance_rating"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
    .to_dict(),
)
print(
    "Train  :",
    train_df["performance_rating"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
    .to_dict(),
)
print(
    "Hidden :",
    hidden_test_df["performance_rating"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
    .to_dict(),
)

# Sanity check: job_satisfaction correlations (should match your signs)
print(
    "\nJob satisfaction correlations (should be negative commute/promo, positive training/remote):"
)
print(
    pd.Series(
        {
            "corr(job_sat, commute_minutes)": np.corrcoef(
                df["job_satisfaction"], df["commute_minutes"]
            )[0, 1],
            "corr(job_sat, training_hours_12mo)": np.corrcoef(
                df["job_satisfaction"], df["training_hours_12mo"]
            )[0, 1],
            "corr(job_sat, remote_work_ratio)": np.corrcoef(
                df["job_satisfaction"], df["remote_work_ratio"]
            )[0, 1],
            "corr(job_sat, last_promotion_years_ago)": np.corrcoef(
                df["job_satisfaction"], df["last_promotion_years_ago"]
            )[0, 1],
        }
    )
    .round(3)
    .to_string()
)

train_path = "data/hris_performance_train.csv"
hidden_path = "data/hris_performance_hidden_test.csv"

if not os.path.exists("data"):
    os.makedirs("data")

train_df.to_csv(train_path, index=False)
hidden_test_df.to_csv(hidden_path, index=False)

print(f"\nSaved: {train_path}  ({len(train_df):,} rows)")
print(f"Saved: {hidden_path} ({len(hidden_test_df):,} rows)")

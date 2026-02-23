from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

SEED = 2026
N_ITER = 25


def load_data():
    # Train Data
    df = pd.read_csv("hris_performance_train.csv")
    y_class = df["performance_rating"]
    X_class = df.drop(columns="performance_rating")

    # Test Data
    test_df = pd.read_csv("hris_performance_hidden_test.csv")
    y_class_test = test_df["performance_rating"]
    X_class_test = test_df.drop(columns="performance_rating")

    return X_class, X_class_test, y_class, y_class_test


def make_rf():
    rf = RandomForestClassifier(
        random_state=SEED,
        bootstrap=True,
        n_jobs=-1,
    )

    param_dist = [
        {
            "n_estimators": randint(100, 1200),
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
            # Pruning knob (log-uniform is the right shape here)
            "ccp_alpha": loguniform(1e-6, 2e-2),
        },
        {
            "n_estimators": randint(100, 1200),
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
            # Leaf smoothing (integer distribution)
            "min_samples_leaf": randint(1, 51),
        },
        {
            "n_estimators": randint(100, 1200),
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
            # Leaf smoothing (integer distribution)
            "min_samples_split": randint(2, 41),
        },
        {
            "n_estimators": randint(100, 1200),
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
            # Depth constraint: sample shallow-to-moderate depths
            "max_depth": randint(3, 41),
        },
    ]

    return rf, param_dist


def make_search(estimator, params: Dict | List[Dict], cv):
    return RandomizedSearchCV(
        estimator,
        params,
        n_iter=N_ITER,
        scoring="f1_weighted",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=SEED,
    )


if __name__ == "__main__":
    np.random.seed(SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    X_class, X_class_test, y_class, y_class_test = load_data()
    rf, param_dist = make_rf()
    search = make_search(rf, param_dist, cv)
    search.fit(X_class, y_class)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best weighted f1 score: {search.best_score_}")
    best_model = search.best_estimator_
    y_hat = best_model.predict(X_class_test)
    print(
        f"Test weighted f1 score: {f1_score(y_class_test, y_hat, average='weighted')}"
    )

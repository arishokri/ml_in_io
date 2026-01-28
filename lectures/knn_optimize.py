import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 2025
SCORE_METHOD = "f1"
INCLUDE_CATEGORICAL = True


def load_and_split(filepath):
    # --- load and split data Split ONCE: keep test set sacred ---
    # X: features df, y: target series (0/1)
    df = pd.read_csv(filepath)
    df.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours"], inplace=True)
    label_map = {"No": 0, "Yes": 1}
    df["Attrition"] = df["Attrition"].map(label_map)
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def compose_search(X_train, include_categorical):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  #! What does this do?
            ("scaler", StandardScaler()),
        ]
    )
    transformers = [("num", numeric, num_cols)]

    if include_categorical:
        cat_cols = [c for c in X_train.columns if c not in num_cols]
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical, cat_cols))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("select", SelectKBest(score_func=mutual_info_classif)),
            ("knn", KNeighborsClassifier()),
        ]
    )

    # --- Search space ---
    # Tip: keep this modest at first; expand later if needed.
    param_grid = {
        # Feature selection (after one-hot, dimensionality can be huge)
        "select__k": [10, 20, 40, 80, 120, 200],
        # k
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 51],
        # Metrics + Minkowski p
        # For 'minkowski': p=1 => Manhattan, p=2 => Euclidean, p~3/4 sometimes helps.
        "knn__metric": ["minkowski", "chebyshev"],
        # p only matters for Minkowski; GridSearchCV will ignore it for chebyshev? (No.)
        # So we split the grid into two dictionaries (recommended) to avoid invalid combos:
    }

    param_grid = [
        {
            "select__k": [5, 10, 15, 20, 23],
            "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["minkowski"],
            "knn__p": [1, 2, 3, 4],
        },
        {
            "select__k": [5, 10, 15, 20, 23],
            "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["chebyshev"],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=SCORE_METHOD,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    return search


def get_best_features(search):
    best_pipe = search.best_estimator_
    prep = best_pipe.named_steps["prep"]
    selector = best_pipe.named_steps["select"]
    feature_names = prep.get_feature_names_out()
    selected_features = feature_names[selector.get_support()]
    return selected_features.tolist()


if __name__ == "__main__":
    print(f"Training for random state: {RANDOM_STATE}")
    print(f"Training for scoring method: {SCORE_METHOD}")
    print(
        "Training", ("WITH" if INCLUDE_CATEGORICAL else "WITHOUT"), "categorical data"
    )
    X_train, X_test, y_train, y_test = load_and_split("ibm_attrition.csv")
    search = compose_search(X_train, INCLUDE_CATEGORICAL)
    search.fit(X_train, y_train)

    print(f"Best CV {SCORE_METHOD}: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    y_pred = search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"TEST accuracy: {test_accuracy:.4f}")
    print(f"TEST f1 score: {test_f1:.4f}")

    print(f"Best Features\n{get_best_features(search)}")

"""Data preparation, descriptive helpers, supervised models, and forecasts."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score, confusion_matrix)

TARGET = "Underemployment Rate Both sexes"
LFPR = "Labor Force Participation Rate Both sexes"
FEATURES = [LFPR, "Labor Force Participation Rate Male",
            "Labor Force Participation Rate Female", "Month"]
SEX_TARGETS = {
    "Both sexes": TARGET,
    "Male": "Underemployment Rate Male",
    "Female": "Underemployment Rate Female",
}
LABELS = ["Low underemployment", "High underemployment"]
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
MODEL_NAMES = ["Naive Bayes", "Logistic Regression", "Decision Tree",
               "Random Forest", "Support Vector Machine"]
FORECAST_MODELS = MODEL_NAMES[1:]


def load_raw_cleaned(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()
    if "Year" not in d or TARGET not in d:
        raise ValueError(f"Descriptive dataset must contain 'Year' and '{TARGET}'. Found: {d.columns.tolist()}")
    # In this supplied descriptive file, Year is coded 1..22 (2005..2026).
    if pd.to_numeric(d["Year"], errors="coerce").max() < 100:
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce") + 2004
    return d


def load_ml(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()
    needed = FEATURES + ["Year", TARGET]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"ML dataset is missing required columns: {missing}")
    d = d.dropna(subset=needed).copy()
    median = float(d[TARGET].median())
    d["Underemployment_Category"] = (d[TARGET] > median).astype(int)
    return d, median


def _continuous_models(seed=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, min_samples_leaf=3),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=2),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
    }


def train_models(df, test_size=0.2, seed=42):
    y = df["Underemployment_Category"]
    # Use a fixed stratified holdout so both classes are represented in test data.
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], y, test_size=test_size,
                                           random_state=seed, stratify=y)
    configs = {}
    # Discretize continuous inputs using training-set quantile cut points only.
    train_cat, test_cat = Xtr.copy(), Xte.copy()
    bin_edges = {}
    for col in FEATURES[:-1]:
        _, edges = pd.qcut(Xtr[col], q=5, retbins=True, duplicates="drop")
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        bin_edges[col] = edges
        train_cat[col] = np.digitize(Xtr[col], edges[1:-1], right=True)
        test_cat[col] = np.digitize(Xte[col], edges[1:-1], right=True)
    train_cat["Month"] = train_cat["Month"].clip(0, 12).astype(int)
    test_cat["Month"] = test_cat["Month"].clip(0, 12).astype(int)
    nb = CategoricalNB().fit(train_cat.astype(int), ytr)
    configs["Naive Bayes"] = (nb, test_cat.astype(int), yte)
    for name, model in _continuous_models(seed).items():
        model.fit(Xtr, ytr)
        configs[name] = (model, Xte, yte)
    return configs


def accuracy_of(configs):
    return {n: accuracy_score(yt, m.predict(Xt)) for n, (m, Xt, yt) in configs.items()}


def comparison_table(configs):
    rows = []
    for name, (model, xt, yt) in configs.items():
        pred = model.predict(xt)
        acc = accuracy_score(yt, pred)
        p, r, f, _ = precision_recall_fscore_support(yt, pred, labels=[0, 1], zero_division=0)
        for i, label in enumerate(LABELS):
            rows.append({"Algorithm": name, "Class": label, "Precision": round(float(p[i]), 4),
                         "Recall": round(float(r[i]), 4), "F1-score": round(float(f[i]), 4),
                         "Accuracy": f"{acc*100:.2f}%" if i == 0 else ""})
    return pd.DataFrame(rows)


def weighted_metrics(model, xt, yt):
    pred = model.predict(xt)
    return {"Accuracy": accuracy_score(yt, pred),
            "Precision": precision_score(yt, pred, average="weighted", zero_division=0),
            "Recall": recall_score(yt, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(yt, pred, average="weighted", zero_division=0)}


def report_table(model, xt, yt):
    pred = model.predict(xt)
    p, r, f, support = precision_recall_fscore_support(yt, pred, labels=[0, 1], zero_division=0)
    out = pd.DataFrame({"Class": LABELS, "Precision": p.round(4), "Recall": r.round(4),
                        "F1-score": f.round(4), "Support": support})
    w = weighted_metrics(model, xt, yt)
    out.loc[len(out)] = ["Weighted average", round(w["Precision"], 4), round(w["Recall"], 4),
                         round(w["F1-score"], 4), int(support.sum())]
    return out


def roc_for(model, xt, yt):
    probs = model.predict_proba(xt)[:, 1]
    fpr, tpr, _ = roc_curve(yt, probs)
    return fpr, tpr, roc_auc_score(yt, probs)


def confusion_for(model, xt, yt):
    return confusion_matrix(yt, model.predict(xt), labels=[0, 1])


def test_predictions(df, model, xt, yt):
    out = df.loc[xt.index, ["Year"] + FEATURES].copy()
    out["Actual"] = yt.map({0: "Low", 1: "High"})
    out["Predicted"] = pd.Series(model.predict(xt), index=xt.index).map({0: "Low", 1: "High"})
    out["Probability high"] = model.predict_proba(xt)[:, 1].round(4)
    out["Correct"] = np.where(out["Actual"] == out["Predicted"], "Yes", "No")
    return out.sort_index()


def pick_forecast_model(configs):
    acc = accuracy_of(configs)
    name = max(FORECAST_MODELS, key=lambda n: acc[n])
    return name, acc[name]


def forecast(df, model_name, years=5):
    """Project numeric underemployment trend and classifier-based high-risk probability.

    These are trend-based exploratory projections, not official labor-force forecasts.
    """
    start = int(df["Year"].max()) + 1
    fut = pd.DataFrame([{"Year": y, "Month": m}
                        for y in range(start, start + years) for m in range(1, 13)])
    time_features = df[["Year", "Month"]].astype(float)
    target_trend = LinearRegression().fit(time_features, df[TARGET])
    fut["Projected_Underemployment_Rate"] = target_trend.predict(fut[["Year", "Month"]])
    fut["Projected_Underemployment_Rate"] = fut["Projected_Underemployment_Rate"].clip(lower=0, upper=100)

    for col in FEATURES[:-1]:
        trend = LinearRegression().fit(time_features, df[col])
        fut[col] = trend.predict(fut[["Year", "Month"]])
    classifier = _continuous_models()[model_name].fit(df[FEATURES], df["Underemployment_Category"])
    fut["Probability_High"] = classifier.predict_proba(fut[FEATURES])[:, 1]
    fut["Predicted_Category"] = (fut["Probability_High"] >= 0.5).astype(int)
    fut["Category_Label"] = fut["Predicted_Category"].map({0: LABELS[0], 1: LABELS[1]})
    annual = fut.groupby("Year").agg(
        Average_Projected_Underemployment=("Projected_Underemployment_Rate", "mean"),
        Average_Probability_High=("Probability_High", "mean"),
        High_Risk_Months=("Predicted_Category", "sum"),
    ).reset_index()
    return fut, annual

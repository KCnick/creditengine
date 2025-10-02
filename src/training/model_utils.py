"""
Reusable model helpers: preprocessing, pipeline, evaluation
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

NUM_FEATURES = ['age','income','credit_lines','delinquencies','utilization','months_active']
CAT_FEATURES = ['region']

def build_pipeline():
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preproc = ColumnTransformer([
        ("num", num_transformer, NUM_FEATURES),
        ("cat", cat_transformer, CAT_FEATURES)
    ])
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=4
    )
    return Pipeline([("preproc", preproc), ("clf", model)])

def evaluate(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    return {"auc": auc, "brier": brier}

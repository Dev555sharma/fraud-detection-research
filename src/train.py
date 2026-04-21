import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    return X_train, y_train, X_test, y_test


def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_scaled, y_train)

    return model, scaler


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


def evaluate(model, X_test, y_test, title, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)

    preds = model.predict(X_test)

    print(f"\n===== {title} =====")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    # Logistic Regression
    model_lr, scaler = train_logistic(X_train, y_train)
    evaluate(model_lr, X_test, y_test, "Logistic Regression", scaler)

    # Random Forest
    model_rf = train_random_forest(X_train, y_train)
    evaluate(model_rf, X_test, y_test, "Random Forest")

    # XGBoost
    model_xgb = train_xgboost(X_train, y_train)
    evaluate(model_xgb, X_test, y_test, "XGBoost")

    # SMOTE + XGBoost
    X_smote, y_smote = apply_smote(X_train, y_train)
    model_xgb_smote = train_xgboost(X_smote, y_smote)
    evaluate(model_xgb_smote, X_test, y_test, "XGBoost + SMOTE")
import shap
import pandas as pd
import xgboost as xgb


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


def run_shap():
    X_train, y_train, X_test, y_test = load_data()

    model = train_model(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:100])  # small sample

    shap.summary_plot(shap_values, X_test[:100])


if __name__ == "__main__":
    run_shap()
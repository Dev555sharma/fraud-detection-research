import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    return X_train, y_train, X_test, y_test


def train_original(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_scaled, y)

    return model, scaler


def train_smote(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_res, y_res)

    return model, scaler


def evaluate(model, scaler, X_test, y_test, title):
    X_test_scaled = scaler.transform(X_test)

    preds = model.predict(X_test_scaled)

    print(f"\n===== {title} =====")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

model_original, scaler1 = train_original(X_train, y_train)
evaluate(model_original, scaler1, X_test, y_test, "WITHOUT SMOTE")

model_smote, scaler2 = train_smote(X_train, y_train)
evaluate(model_smote, scaler2, X_test, y_test, "WITH SMOTE")
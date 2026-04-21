import pandas as pd
from imblearn.over_sampling import SMOTE


def load_data():
    train = pd.read_csv("data/processed/train.csv")

    X = train.drop("target", axis=1)
    y = train["target"]

    return X, y


def apply_smote(X, y):
    print("\n===== BEFORE SMOTE =====")
    print(y.value_counts())

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\n===== AFTER SMOTE =====")
    print(pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled


if __name__ == "__main__":
    X, y = load_data()
    X_res, y_res = apply_smote(X, y)
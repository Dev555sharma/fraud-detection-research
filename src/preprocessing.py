import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"


def load_data(path="data/raw/creditcard.csv"):
    df = pd.read_csv(path)
    return df


def basic_eda(df):
    print("\n===== BASIC DATA ANALYSIS =====")

    print("\nDataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nClass Distribution:")
    print(df[TARGET].value_counts())

    print("\nClass Ratio:")
    print(df[TARGET].value_counts(normalize=True))


def split_data(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,   # VERY IMPORTANT
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    train = X_train.copy()
    train["target"] = y_train

    test = X_test.copy()
    test["target"] = y_test

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    df = load_data()

    basic_eda(df)

    X_train, X_test, y_train, y_test = split_data(df)

    save_data(X_train, X_test, y_train, y_test)

    print("\n✅ Data preprocessing complete")
"""Task 5: Personal Loan Acceptance Prediction."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


BANK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional/bank-additional-full.csv"
)


def load_data():
    """Load bank marketing data or create fallback dataset."""
    try:
        df = pd.read_csv(BANK_URL, sep=";")
        print("Loaded bank marketing dataset from URL.")
    except Exception:
        print("Could not load bank dataset from URL. Generating synthetic dataset.")
        df = pd.DataFrame(
            {
                "age": [30, 40, 50, 35, 28],
                "job": ["admin.", "technician", "entrepreneur", "blue-collar", "services"],
                "marital": ["married", "single", "married", "divorced", "single"],
                "education": ["tertiary", "secondary", "tertiary", "secondary", "primary"],
                "default": ["no", "no", "yes", "no", "no"],
                "housing": ["yes", "yes", "no", "yes", "no"],
                "loan": ["no", "yes", "no", "no", "no"],
                "contact": ["cellular", "telephone", "cellular", "cellular", "unknown"],
                "month": ["may", "jun", "jul", "aug", "may"],
                "duration": [100, 200, 150, 120, 90],
                "campaign": [1, 2, 1, 1, 3],
                "pdays": [999, 999, 999, 999, 999],
                "previous": [0, 0, 0, 0, 0],
                "poutcome": ["unknown", "unknown", "unknown", "unknown", "unknown"],
                "emp.var.rate": [1.1, 1.4, 1.1, 1.4, 1.1],
                "cons.price.idx": [93.994, 94.465, 93.200, 94.055, 93.200],
                "cons.conf.idx": [-36.4, -41.8, -36.4, -41.8, -36.4],
                "euribor3m": [4.857, 4.963, 4.021, 4.857, 4.021],
                "nr.employed": [5191, 5228, 5195, 5228, 5195],
                "y": ["no", "yes", "no", "no", "yes"],
            }
        )
    return df


def clean_data(df: pd.DataFrame):
    """Encode categorical columns and prepare features."""
    categorical_cols = df.select_dtypes(include=["object"]).columns.drop("y")
    encoder = OneHotEncoder(drop="first", sparse=False)
    encoded = pd.DataFrame(
        encoder.fit_transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )
    df_model = pd.concat([df.drop(columns=categorical_cols), encoded], axis=1)
    df_model["target"] = df["y"].map({"yes": 1, "no": 0})
    X = df_model.drop(columns=["y", "target"])
    y = df_model["target"]
    return X, y


def plot_split_by_groups(df: pd.DataFrame):
    """Plot loan acceptance by job and marital status."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x="job", hue="y", data=df)
    plt.xticks(rotation=45)
    plt.title("Loan Offer Acceptance by Job")
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.countplot(x="marital", hue="y", data=df)
    plt.title("Loan Offer Acceptance by Marital Status")
    plt.show()


def train_model(X: pd.DataFrame, y: pd.Series):
    """Train and evaluate a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred))
    return model


def main():
    df = load_data()
    print(df.head())
    plot_split_by_groups(df)
    X, y = clean_data(df)
    train_model(X, y)


if __name__ == "__main__":
    main()

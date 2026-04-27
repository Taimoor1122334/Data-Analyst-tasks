"""Task 2: Credit Risk Prediction."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


LOAN_URL = (
    "https://raw.githubusercontent.com/dphi-official/Datasets/master/loan_default_prediction.csv"
)


def load_data():
    """Load loan data from URL or create a synthetic dataset if unavailable."""
    try:
        df = pd.read_csv(LOAN_URL)
        print("Loaded loan dataset from URL.")
    except Exception:
        print("Could not load loan dataset from URL. Generating synthetic dataset.")
        df = pd.DataFrame(
            {
                "loan_amount": [5000, 10000, 15000, 20000, 25000, 30000],
                "term": [36, 60, 36, 60, 36, 60],
                "education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate", "Graduate"],
                "income": [4000, 6000, 3000, 9000, 5000, 7000],
                "credit_score": [700, 660, 720, 680, 640, 710],
                "defaulted": [0, 1, 0, 0, 1, 0],
            }
        )
    return df


def clean_data(df: pd.DataFrame):
    """Handle missing values and encode categorical columns."""
    df = df.dropna().copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.factorize(df[col])[0]
    return df


def get_target_column(df: pd.DataFrame):
    """Identify the most likely target column for credit risk."""
    for candidate in ["defaulted", "loan_status", "Loan_Status", "target", "default"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No target column found in loan dataset.")


def plot_feature_relationships(df: pd.DataFrame, target: str):
    """Plot key feature relationships for loan risk."""
    plt.figure(figsize=(10, 6))
    if "income" in df.columns and "loan_amount" in df.columns:
        sns.scatterplot(x="income", y="loan_amount", hue=target, data=df)
        plt.title("Loan Amount vs Income")
        plt.show()

    if "education" in df.columns and "loan_amount" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x="education", hue=target, data=df)
        plt.title("Default by Education")
        plt.show()


def train_model(df: pd.DataFrame, target: str):
    """Train a logistic regression model."""
    X = df.drop(columns=[target])
    y = df[target]
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
    df = clean_data(df)
    target = get_target_column(df)
    print("Target column:", target)
    print(df.head())
    plot_feature_relationships(df, target)
    train_model(df, target)


if __name__ == "__main__":
    main()

"""Task 3: Customer Churn Prediction."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


CHURN_URL = "https://raw.githubusercontent.com/sharmaroshan/Customer-Churn-Prediction/master/Churn_Modelling.csv"


def load_data():
    """Load churn modeling data or generate synthetic fallback data."""
    try:
        df = pd.read_csv(CHURN_URL)
        print("Loaded churn dataset from URL.")
    except Exception:
        print("Could not load churn dataset from URL. Generating synthetic dataset.")
        df = pd.DataFrame(
            {
                "RowNumber": [1, 2, 3, 4, 5],
                "CustomerId": [15634602, 15647311, 15619304, 15701354, 15737888],
                "Surname": ["Hargrave", "Hill", "Onio", "Boni", "Mitchell"],
                "CreditScore": [619, 608, 502, 699, 850],
                "Geography": ["France", "Spain", "France", "Spain", "France"],
                "Gender": ["Female", "Female", "Female", "Female", "Male"],
                "Age": [42, 41, 42, 39, 43],
                "Tenure": [2, 1, 8, 1, 2],
                "Balance": [0.0, 83807.86, 159660.8, 0.0, 125510.82],
                "NumOfProducts": [1, 1, 3, 2, 1],
                "HasCrCard": [1, 0, 1, 0, 1],
                "IsActiveMember": [1, 1, 0, 0, 1],
                "EstimatedSalary": [101348.88, 112542.58, 113931.57, 93826.63, 79084.1],
                "Exited": [1, 0, 1, 0, 1],
            }
        )
    return df


def clean_data(df: pd.DataFrame):
    """Prepare data by dropping identifiers and encoding categorical variables."""
    df = df.drop(columns=[col for col in ["RowNumber", "CustomerId", "Surname"] if col in df.columns], errors="ignore")
    target = "Exited"
    categorical = [col for col in ["Geography", "Gender"] if col in df.columns]
    if categorical:
        encoder = OneHotEncoder(drop="first", sparse=False)
        encoded = pd.DataFrame(
            encoder.fit_transform(df[categorical]),
            columns=encoder.get_feature_names_out(categorical),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=categorical), encoded], axis=1)
    return df, target


def plot_overview(df: pd.DataFrame, target: str):
    """Plot churn distribution and a sample feature."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target, data=df)
    plt.title("Churn Distribution")
    plt.show()
    if "Age" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="Age", hue=target, kde=True)
        plt.title("Age Distribution by Churn")
        plt.show()


def train_model(df: pd.DataFrame, target: str):
    """Train a Random Forest classifier and print evaluation metrics."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred))
    return model, X.columns


def show_feature_importance(model, feature_names):
    """Display feature importance for the trained model."""
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("Top feature importances:\n", importance.head(10))
    plt.figure(figsize=(10, 6))
    importance.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances")
    plt.show()


def main():
    df = load_data()
    df, target = clean_data(df)
    print(df.head())
    plot_overview(df, target)
    model, feature_names = train_model(df, target)
    show_feature_importance(model, feature_names)


if __name__ == "__main__":
    main()

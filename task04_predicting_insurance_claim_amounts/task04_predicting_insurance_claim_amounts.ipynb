"""Task 4: Predicting Insurance Claim Amounts."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


INSURANCE_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


def load_data():
    """Load insurance data or generate a synthetic fallback dataset."""
    try:
        df = pd.read_csv(INSURANCE_URL)
        print("Loaded insurance dataset from URL.")
    except Exception:
        print("Could not load insurance dataset from URL. Generating synthetic dataset.")
        df = pd.DataFrame(
            {
                "age": [19, 18, 28, 33, 32, 31],
                "sex": ["female", "male", "male", "male", "male", "female"],
                "bmi": [27.9, 33.77, 33.0, 22.705, 28.88, 25.74],
                "children": [0, 1, 3, 0, 0, 0],
                "smoker": ["yes", "no", "no", "no", "no", "no"],
                "region": ["southwest", "southeast", "southeast", "northwest", "northwest", "southeast"],
                "charges": [16884.92, 1725.55, 4449.46, 21984.47, 3866.86, 3756.62],
            }
        )
    return df


def clean_data(df: pd.DataFrame):
    """Encode categorical features and prepare the model dataset."""
    df = pd.get_dummies(df, columns=[col for col in ["sex", "smoker", "region"] if col in df.columns], drop_first=True)
    X = df.drop(columns=["charges"])
    y = df["charges"]
    return X, y


def plot_relationships(df: pd.DataFrame):
    """Plot how age, BMI, and smoking status relate to charges."""
    if {"age", "charges", "smoker_yes"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="age", y="charges", hue="smoker_yes", data=df)
        plt.title("Insurance Charges by Age and Smoking Status")
        plt.show()
    if {"bmi", "charges", "smoker_yes"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="bmi", y="charges", hue="smoker_yes", data=df)
        plt.title("Insurance Charges by BMI and Smoking Status")
        plt.show()


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Train linear regression and report MAE and RMSE."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print("MAE:", mae)
    print("RMSE:", rmse)
    return model


def main():
    df = load_data()
    print(df.head())
    plot_relationships(df)
    X, y = clean_data(df)
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()

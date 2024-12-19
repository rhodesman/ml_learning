from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df, target_col="price_change"):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column name for the target variable.

    Returns:
        tuple: Training, validation, and test sets (features and labels).
    """
    features = df.drop(columns=[target_col, "time", "date"])  # Drop non-numeric columns
    target = df[target_col]

    # Split into training + validation (80%) and test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.2, random_state=42)

    # Split the temp set into validation (50% of temp) and test (50% of temp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_classifier(X_train, y_train, X_val, y_val):
    """
    Train a binary classification model and evaluate it.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        model: Trained model.
    """

    # Debugging: Check for missing values
    print("Missing values per column in training set:")
    print(X_train.isna().sum())
    print("Missing values per column in validation set:")
    print(X_val.isna().sum())

    # Preprocess features to handle missing values
    X_train = preprocess_features(X_train)
    X_val = preprocess_features(X_val)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_pred))

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
    """
    y_pred = model.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred))

from sklearn.impute import SimpleImputer

def preprocess_features(X):
    """
    Preprocess feature matrix by imputing missing values.

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        pd.DataFrame: Preprocessed feature matrix.
    """

    # Drop columns with all NaN values
    X = X.dropna(axis=1, how="all")

    print("Shape of X after dropping all-NaN columns:", X.shape)

    # Impute missing values with the mean of each column
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)

if __name__ == "__main__":
    # Load processed data
    merged_data = pd.read_csv("data/processed/merged_data.csv")

    # Add price change labels
    merged_data = add_price_change_label(merged_data)

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_data, target_col="price_change")

    # Train the classifier
    model = train_classifier(X_train, y_train, X_val, y_val)

    # Evaluate on test data
    evaluate_model(model, X_test, y_test)
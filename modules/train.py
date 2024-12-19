from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pandas as pd

def split_data(df, target_col):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        Tuple of DataFrames: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Dynamically drop columns that exist in the DataFrame
    drop_columns = [col for col in ["time", "date", target_col, "ticker"] if col in df.columns]
    print(f"Dropping columns: {drop_columns}")  # Debugging print

    # Features and target
    features = df.drop(columns=drop_columns)  # Drop non-numeric columns
    target = df[target_col]

    # Remove non-numeric columns from features
    numeric_features = features.select_dtypes(include=["number"])
    print(f"Numeric features retained: {numeric_features.columns}")  # Debugging print

    # Train-validation-test split
    X_train, X_temp, y_train, y_temp = train_test_split(numeric_features, target, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_classifier(X_train, y_train, X_val, y_val):
    """
    Train a binary classification model and evaluate it.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        model, list: Trained model and list of feature names.
    """
    # Preprocess training data and save feature names
    X_train, feature_names = preprocess_features(X_train)

    # Preprocess validation data with the same features
    X_val, _ = preprocess_features(X_val, feature_names=feature_names)

    # Use class balancing
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", scores.mean())

    # Train final model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_pred, zero_division=0))

    return model, feature_names

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model on the test set.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        feature_names (list): List of feature names used during training.
    """
    # Preprocess test data with training feature names
    X_test, _ = preprocess_features(X_test, feature_names=feature_names)

    # Make predictions
    y_pred = model.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred, zero_division=0))

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest classifier and evaluate it on the validation set.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost classifier and evaluate it on the validation set.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model.
    """
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    return model

def preprocess_features(X, feature_names=None):
    """
    Preprocess feature matrix by imputing missing values and ensuring consistent features.

    Args:
        X (pd.DataFrame): Feature matrix.
        feature_names (list or None): Optional list of feature names to ensure consistency.

    Returns:
        pd.DataFrame, list: Preprocessed feature matrix and list of feature names.
    """
    #print("Feature names used for training:")
    #print(feature_names)

    # Drop columns with all NaN values
    X = X.dropna(axis=1, how="all")

    # Impute missing values with the mean of each column
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Convert back to DataFrame
    X_processed = pd.DataFrame(X_imputed, columns=X.columns)

    # If feature_names is provided, enforce consistent columns
    if feature_names:
        missing_features = [f for f in feature_names if f not in X_processed.columns]
        for f in missing_features:
            X_processed[f] = 0  # Add missing features as zeros
        X_processed = X_processed[feature_names]

    #print("Feature names after preprocessing:", X.columns.tolist())
    return X_processed, X.columns.tolist()

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
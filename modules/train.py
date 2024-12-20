from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
import numpy as np

def split_data(df, target_col):
    """
    Splits the data into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The full dataset.
        target_col (str): The target column for prediction.

    Returns:
        tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test, encoders, tickers_test).
    """
    print("Dropping columns:", [target_col, "time", "date"])

    # Initialize label encoders
    crypto_encoder = LabelEncoder()
    stock_encoder = LabelEncoder()

    # Encode ticker columns
    if "ticker_crypto" in df.columns:
        df["ticker_crypto_encoded"] = crypto_encoder.fit_transform(df["ticker_crypto"])
    else:
        print("Warning: 'ticker_crypto' column not found in dataset.")
        df["ticker_crypto_encoded"] = None

    if "ticker_stock" in df.columns:
        df["ticker_stock_encoded"] = stock_encoder.fit_transform(df["ticker_stock"])
    else:
        print("Warning: 'ticker_stock' column not found in dataset.")
        df["ticker_stock_encoded"] = None

    # Features and target
    try:
        features = df.drop(columns=[target_col, "time", "date", "ticker_crypto", "ticker_stock"], errors="ignore").copy()
    except KeyError as e:
        print(f"Error dropping columns: {e}")
        features = df.copy()  # Use the entire dataset if specific columns are missing

    target = df[target_col] if target_col in df.columns else None
    if target is None:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Columns in X_train:", X_train.columns)
    print("Columns in X_val:", X_val.columns)
    print("Columns in X_test:", X_test.columns)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # Capture ticker information for test set
    tickers_test = df.loc[X_test.index, ["ticker_crypto_encoded", "ticker_stock_encoded"]]

    return X_train, X_val, X_test, y_train, y_val, y_test, {"crypto": crypto_encoder, "stock": stock_encoder}, tickers_test

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

    # Check for non-numeric columns
    if not all(np.issubdtype(dtype, np.number) for dtype in X_train.dtypes):
        print("Non-numeric columns in X_train:")
        print(X_train.select_dtypes(exclude=[np.number]).head())
        raise ValueError("X_train contains non-numeric data. Please check feature preparation.")

    if not all(np.issubdtype(dtype, np.number) for dtype in X_val.dtypes):
        print("Non-numeric columns in X_val:")
        print(X_val.select_dtypes(exclude=[np.number]).head())
        raise ValueError("X_val contains non-numeric data. Please check feature preparation.")

    # Debugging step: Inspect the first few rows of training features
    print("Sample of X_train:")
    print(X_train.head())

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error while training Random Forest: {e}")
        raise

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    # Extract and save feature importances
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    importance_file = "data/processed/random_forest_feature_importances.csv"
    importance_df.to_csv(importance_file, index=False)
    print(f"Random Forest feature importances saved to {importance_file}")

    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost classifier, evaluate it on the validation set, 
    and inspect feature importance.

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
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    # Inspect feature importance
    print("\nExtracting Feature Importances...")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance Summary:")
    print(importance_df.head())

    # Save feature importances for further analysis
    importance_file = "data/processed/xgboost_feature_importances.csv"
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importances saved to {importance_file}")

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

def ensemble_predict(rf_model, xgb_model, X_test):
    """
    Generate predictions using an ensemble of Random Forest and XGBoost models.

    Args:
        rf_model: Trained Random Forest model.
        xgb_model: Trained XGBoost model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Ensemble predictions (0 or 1).
    """
    print("Generating ensemble predictions...")
    
    # Random Forest predictions
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of class 1
    # XGBoost predictions
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Average the probabilities
    ensemble_pred_proba = (rf_pred_proba + xgb_pred_proba) / 2

    # Convert probabilities to binary predictions
    ensemble_pred = np.where(ensemble_pred_proba >= 0.5, 1, 0)
    
    return ensemble_pred

def train_model_by_asset_type(data, model_type="RandomForest"):
    """Train models for each asset type."""
    results = {}

    for asset in data["ticker"].unique():
        asset_data = data[data["ticker"] == asset]
        X_train, X_val, y_train, y_val = split_data(asset_data)

        if model_type == "RandomForest":
            model = train_random_forest(X_train, y_train, X_val, y_val)
        elif model_type == "XGBoost":
            model = train_xgboost(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        results[asset] = model

    return results

def train_model(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load processed data
    merged_data = pd.read_csv("data/processed/merged_data.csv")

    # Add price change labels
    merged_data = add_price_change_label(merged_data)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, tickers_test = split_data(merged_data, target_col="price_change")

    # Train the classifier
    model = train_classifier(X_train, y_train, X_val, y_val)

    # Evaluate on test data
    evaluate_model(model, X_test, y_test)
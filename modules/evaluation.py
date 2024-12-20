from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_pred))
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))

def save_feature_importance(model, filepath):
    feature_importances = pd.DataFrame({
        'Feature': X_val.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importances.to_csv(filepath, index=False)
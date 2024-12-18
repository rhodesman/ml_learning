from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report


class Classifier:
    def __init__(self, model_type="logistic"):
        if model_type == "logistic":
            self.model = LogisticRegression()
        elif model_type == "rf":
            self.model = RandomForestClassifier()
        elif model_type == "xgb":
            self.model = XGBClassifier()
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(force_col_wise=True)
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report
    

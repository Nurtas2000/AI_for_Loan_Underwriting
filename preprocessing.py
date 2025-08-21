import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class LoanDataPreprocessor:
    def __init__(self):
        self.numeric_features = ['age', 'income', 'employment_length', 'credit_score', 
                               'debt_to_income', 'loan_amount', 'loan_term', 'recent_inquiries']
        self.categorical_features = ['home_ownership', 'loan_purpose']
        self.preprocessor = self._create_preprocessor()
        
    def _create_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)])
    
    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X):
        return self.preprocessor.transform(X)

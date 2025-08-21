from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate
)
import pandas as pd

class FairnessAnalyzer:
    def __init__(self, sensitive_features):
        self.sensitive_features = sensitive_features
    
    def analyze(self, y_true, y_pred):
        results = {}
        
        for feature in self.sensitive_features:
            sr = selection_rate(y_true, y_pred, sensitive_features=feature)
            dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=feature)
            eod = equalized_odds_difference(y_true, y_pred, sensitive_features=feature)
            
            results[feature] = {
                'selection_rate': sr,
                'demographic_parity_difference': dpd,
                'equalized_odds_difference': eod
            }
        
        return pd.DataFrame(results).T

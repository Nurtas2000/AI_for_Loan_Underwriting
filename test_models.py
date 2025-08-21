import pytest
import pandas as pd
import numpy as np
from src.models.train import train_model
from src.data.preprocessing import LoanDataPreprocessor

@pytest.fixture
def sample_data():
    data = {
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'credit_score': [650, 700, 750],
        'debt_to_income': [0.2, 0.3, 0.4],
        'home_ownership': ['Rent', 'Own', 'Mortgage'],
        'defaulted': [0, 1, 0]
    }
    return pd.DataFrame(data)

def test_model_training(sample_data):
    X = sample_data.drop('defaulted', axis=1)
    y = sample_data['defaulted']
    
    model, metrics = train_model(X, y)
    
    assert model is not None
    assert 'roc_auc' in metrics
    assert metrics['roc_auc'] >= 0.5

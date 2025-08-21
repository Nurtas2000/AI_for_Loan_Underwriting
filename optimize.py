import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from ..data.preprocessing import LoanDataPreprocessor

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }
    
    preprocessor = LoanDataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_processed, y_train,
              eval_set=[(X_val_processed, y_val)],
              early_stopping_rounds=50,
              verbose=False)
    
    y_pred = model.predict_proba(X_val_processed)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    return auc

def run_optimization(X_train, y_train, X_val, y_val, n_trials=50):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                  n_trials=n_trials)
    return study.best_params

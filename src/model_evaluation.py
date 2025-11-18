import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise e
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise e
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise e
def load_model(file_path):
    try:
         
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise e
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise e
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
def evaluatiom_data(model,X_test,y_test):
    try:
        y_pred=model.predict(X_test)
        y_pred_proba=model.predict_proba(X_test)[:,1]
        accuracy=accuracy_score(y_pred,y_test)
        precision=precision_score(y_pred,y_test)
        recall=recall_score(y_pred,y_test)
        auc=roc_auc_score(y_test,y_pred_proba)
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug("Metrics are evaluated")
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise
def save_metrics(file_path,metrics_dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics_dict,file,indent=4)
        logger.debug("Metrics saved to  %s",file_path)
    except Exception as e:
        logger.error("Unexpected error while saving metrics: %s",e)
def main():
    try:
        params=load_params("params.yaml")
        clf=load_model("./models/model.pkl")
        test_data=load_data("./data/vectorized/test_vectorized.csv")
        X_test=test_data.iloc[:,:-1]
        y_test=test_data.iloc[:,-1]
        metrics_dict=evaluatiom_data(clf,X_test,y_test)
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)
        save_metrics('reports/metrics.json',metrics_dict)
    except Exception as e:
        logger.debug("Unexpexted error occur in model evaluation: %s",e)
        raise e
if __name__=='__main__':
    main()


        
import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import ExtraTreesClassifier
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,"model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

Formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(Formatter)
file_handler.setFormatter(Formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(param_url):
    try:
        with open(param_url,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Params are loaded in %s", param_url)
        return params
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise e
    except Exception as e:
        logger.error("Unexpected error occur while loading params: %s", e)
        raise e
def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data is loaded")
        return df
    except pd.errors.ParserError as e:
        logger.error("Parsing error occur: %s", e)
        raise e
    except FileNotFoundError as e:
        logger.error("File not found in %s",e)
        raise e
    except Exception as e:
        logger.error("Unexpected error occur while loading data: %s",e)
        raise e
def train_model(X_train,y_train,n_estimators,random_state):
    try:
        if (X_train.shape[0]!=y_train.shape[0]):
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        logger.debug(f"Initialising ExtraTreesClassifiers model with parameters n_estimaor={n_estimators} amd random_state={random_state}")
        clf=ExtraTreesClassifier(n_estimators=n_estimators,random_state=random_state)
        clf.fit(X_train,y_train)
        logger.debug("Model trianing completed")
        return clf
    except ValueError as e:
        logger.error("Value Error occured: %s",e)
        raise e
    except Exception as e:
        logger.error("Unexpected error occur while traiining: %s",e)
        raise e
def save_model(model,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file=file)
        logger.debug("Model saved to %s,",file_path)
    except FileNotFoundError as e:
        logger.error("Filepath not found: %s",file_path)
        raise e
    except Exception as e:
        logger.error("Unexpected error occur while saving the data: %s",e)
        raise e
def main():
    try:
        params=load_param("params.yaml")
        n_estimators=params['model_building']['n_estimators']
        random_state=params['model_building']['random_state']
        data_path1="./data/vectorized/train_vectorized.csv"
        train_data=load_data(data_path1)
        clf=train_model(train_data.drop(columns=['label']),train_data['label'],n_estimators,random_state)
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        raise e

if __name__ == '__main__':
    main()






import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger()
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_path_file=os.path.join(log_dir,'featuring_engineering.log')
file_handler=logging.FileHandler(log_path_file)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_url: str)->dict:
    try:
        with open(params_url,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Params are loaded",params_url)
        return params
    except yaml.YAMLError as e:
        logger.error("Yaml error: %s",e)
        raise e
    except Exception as e:
        logger.error("Unexpected error occur: %s",e)
        raise e

def load_data(file_path: str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("Data loaded and NaNs filled from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_df,test_df,max_features):
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)
        X_train=train_df['text'].values
        y_train=train_df['target'].values
        X_test=test_df['text'].values
        y_test=test_df['target'].values
        X_train_bow=vectorizer.fit_transform(X_train)
        X_test_bow=vectorizer.transform(X_test)
        train_data=pd.DataFrame(X_train_bow.toarray())
        train_data['label']=y_train
        test_data=pd.DataFrame(X_test_bow.toarray())
        test_data['label']=y_test
        logger.debug("tfdif applied and data transformed")
        return train_data,test_data
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise e
def save_data(df,filepath):
    try:
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        df.to_csv(filepath,index=False)
        logger.debug("File has been saved in %s",filepath)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise e
def main( ):
    try:
        params=load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        train_vectorized_df,test_vectorized_df=apply_tfidf(train_data,test_data,max_features)
        file_path1="./data/vectorized/train_vectorized.csv"
        file_path2="./data/vectorized/test_vectorized.csv"
        save_data(train_vectorized_df,file_path1)
        save_data(test_vectorized_df,file_path2)
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        raise e
if __name__=='__main__':
    main()


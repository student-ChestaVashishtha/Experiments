import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# =========================
#       LOGGING
# =========================

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================
#       FUNCTIONS
# =========================

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise FileNotFoundError(f"File not found: {params_path}")

    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise e

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise e


def load_data(data_url: str) -> pd.DataFrame:
    try:
        # The spam dataset contains special characters → latin-1 needed
        df = pd.read_csv(data_url, encoding='latin-1')
        logger.debug("Data loaded successfully from %s", data_url)
        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise e

    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise e


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Drop extra unnamed columns safely
        unwanted = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
        df.drop(columns=[c for c in unwanted if c in df.columns], inplace=True)

        df.rename(columns={'label': 'target', 'v2': 'text'}, inplace=True)

        logger.debug("Data preprocessing completed")
        return df

    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise e


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_dir = os.path.join(data_path, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        train_data.to_csv(os.path.join(raw_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_dir, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_dir)

    except Exception as e:
        logger.error("Error while saving data: %s", e)
        raise e


# =========================
#         MAIN
# =========================

def main():
    try:
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        # FIXED DATASET URL (public working link)
        data_url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/spam.csv"

        df = load_data(data_url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        save_data(train_data, test_data, data_path="./data")

    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        raise e


if __name__ == "__main__":
    main()

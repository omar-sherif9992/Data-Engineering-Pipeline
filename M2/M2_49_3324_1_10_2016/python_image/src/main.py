import os
from ingestion import ingest
import sys

ROOT_DIR = os.path.abspath(".")
DATASET_DIR = f"{ROOT_DIR}/data"  # Directory for storing datasets.

year = 2016
month = 10
cleaned_dataset_file_name = f"green_trip_data_{year}-{month}clean.csv"
lookup_table_file_name = "lookup_table_green_taxis.csv"


if __name__ == "__main__":
    cleaned_file_path = os.path.join(DATASET_DIR, cleaned_dataset_file_name)
    lookup_table_file_path = os.path.join(DATASET_DIR, lookup_table_file_name)
    
    if not os.path.exists(cleaned_file_path) or not os.path.exists(lookup_table_file_path):
        print("Preprocessing the dataset...")
        try:
            print("Trying to run python3...")
            os.system("python3 src/preprocessing.py")
        except Exception as e:
            print("Trying to run python...")
            os.system("python src/preprocessing.py")
        print("Preprocessing completed.")
    
    print("Ingesting the dataset...")
    
    ingest([cleaned_file_path, lookup_table_file_path], ["green_taxi_10_2016", "lookup_green_taxi_10_2016"])
    sys.stdout.flush()

        
    

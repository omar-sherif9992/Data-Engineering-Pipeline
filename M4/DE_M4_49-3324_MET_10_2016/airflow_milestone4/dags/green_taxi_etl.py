
import pandas as pd
import numpy as np
# For Label Encoding
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

from create_dashboard import create_dashboard_ui
from ingestion import ingest
from preprocessing import clean_missing,encoding,extract_gps
import os

def extract_clean(filename):
    target_path = f'/opt/airflow/data/green_tripdata_2016-10_clean.csv'
    if (os.path.exists(target_path)):
        print('file already exists no need to extract and clean')
        return
    df = pd.read_csv(filename)
    df = clean_missing(df)
    
    df.to_csv(target_path,index=False)
    print('loaded after cleaning succesfully')
    
    
def encode_load(filename):
    target_path = f'/opt/airflow/data/green_tripdata_2016-10_transformed.csv'
    if (os.path.exists(target_path) and os.path.exists('/opt/airflow/data/lookup_table.csv')):
        print('file already exists no need to encode and load')
        return
    try:
        df = pd.read_csv(filename)
        df = encoding(df,'/opt/airflow/data/lookup_table.csv')
        df.to_csv(target_path,index=False)
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')


def extract_additional_resources(filename):
    target_path ='/opt/airflow/data/green_tripdata_2016-10_integrated.csv'
    if (os.path.exists(target_path)):
        print('file already exists no need to extract additional resources')
        return


    df = pd.read_csv(filename)

    extract_gps(df,'/opt/airflow/data/locations.csv')
    df.to_csv(target_path, index=False)



def load_to_postgres(filename,lookup_file_name):
    ingest([filename,lookup_file_name],['M4_green_taxis_10_2016','lookup_table'])



def create_dashboard(filename):
    return create_dashboard_ui(filename)


#load_to_postgres('../data/green_tripdata_10_2016_transformed.csv','../data/lookup_table.csv')


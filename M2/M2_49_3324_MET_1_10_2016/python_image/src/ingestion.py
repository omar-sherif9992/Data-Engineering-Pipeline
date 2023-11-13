import pandas as pd

from sqlalchemy import create_engine

import sys

def ingest(file_paths:list[str],file_names:list[str]):
    """
    ingest data from pandas to postgresql
    """
    engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxi')

    if(engine.connect()):
        print('connected successfully')
    else:
        print('failed to connect')
    
    print('ingestion started')
    
    
    mode = 'fail'
    for file_path,file_name in zip(file_paths,file_names):
        try: 
            print(f'reading {file_name} table')
            df = pd.read_csv(file_path)
            print(f'ingesting {file_name} table')        
            df.to_sql(name = file_name,con = engine,if_exists=mode)
            print(f'{file_name} table created successfully')
        except Exception as e:
            print(f'creating table failed because it already exists: {e}')
            sys.stdout.flush()
       
    print('ingestion completed')
    sys.stdout.flush()


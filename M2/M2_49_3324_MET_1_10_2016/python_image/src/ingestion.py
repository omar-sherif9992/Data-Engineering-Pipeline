import pandas as pd

from sqlalchemy import create_engine



def ingest(file_paths:list[str],file_names:list[str]):
    """
    ingest data from pandas to postgresql
    """
    engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxi')

    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')

    for file_path,file_name in zip(file_paths,file_names):
        df = pd.read_csv(file_path)
        print(df.head(1))
        
        df.to_sql(name = file_name,con = engine,if_exists='replace')

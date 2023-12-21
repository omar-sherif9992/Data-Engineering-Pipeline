from green_taxi_etl import extract_clean, load_to_postgres, create_dashboard, encode_load, extract_additional_resources
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": True,
    "start_date": days_ago(1),
    "retries": 1,
}

dag = DAG(
    "green_taxi_etl_pipeline",
    default_args=default_args,
    description="green_taxi etl pipeline",
)
with DAG(
    dag_id="green_taxi_etl_pipeline",
    schedule_interval="@once",
    default_args=default_args,
    tags=["green_taxi-pipeline"],
) as dag:
    extract_clean_task = PythonOperator(
        task_id="extract_dataset",
        python_callable=extract_clean,
        op_kwargs={"filename": "/opt/airflow/data/green_tripdata_2016-10.csv"},
    )
    encoding_load_task = PythonOperator(
        task_id="encoding",
        python_callable=encode_load,
        op_kwargs={"filename": "/opt/airflow/data/green_tripdata_2016-10_clean.csv"},
    )
    extract_additional_resources_task = PythonOperator(
        task_id="extract_additional_resources",
        python_callable=extract_additional_resources,
        op_kwargs={"filename": "/opt/airflow/data/green_tripdata_2016-10_transformed.csv"},
    )
    load_to_postgres_task = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres,
        op_kwargs={
            "filename": "/opt/airflow/data/green_tripdata_2016-10_integrated.csv",
            "lookup_file_name": "/opt/airflow/data/lookup_table.csv",
        },
    )
    create_dashboard_task = PythonOperator(
        task_id="create_dashboard_task",
        python_callable=create_dashboard,
        op_kwargs={"filename": "/opt/airflow/data/green_tripdata_2016-10_clean.csv"},
    )

    extract_clean_task >> encoding_load_task >> extract_additional_resources_task >> load_to_postgres_task >> create_dashboard_task

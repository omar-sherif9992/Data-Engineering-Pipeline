### Initial Setup 
 - mkdir -p ./dags ./logs ./plugins ./data
 - echo -e "AIRFLOW_UID=$(id -u)" > .env
 
 
### Initialise airflow
- sudo docker-compose build 
- sudo docker compose up airflow-init
- sudo docker-compose up -d --remove-orphans


### Credentials
- username: airflow
- password: airflow


### To discover Networks:
- sudo docker network ls


### to refresh the airflow
- sudo docker compose up --build --remove-orphans
- sudo docker-compose down --volumes --remove-orphans



## to check for the database in the terminal
    
- run the following command to check the volumes
```
sudo docker volume ls
```

- find the volume name of the postgres container
```
postgres_taxis_datasets_green_taxi_10_2016_postgres
```

- run another postgres container
```
sudo docker run -it --rm --name postgres_container -v postgres_taxis_datasets_green_taxi_10_2016_postgres:/var/lib/postgresql/data -e  POSTGRES_USER=root -e POSTGRES_PASSWORD=root  -e POSTGRES_DB=green_taxi -p 5432:5432 postgres:13
```

- open another terminal and run the following command to get into the postgres container
```
sudo docker exec -it postgres_container bash
```

- run the following command to get into the postgres database
```
 psql -d green_taxi_etl
```

- run the following command to check the tables in the database
```
\dt
```



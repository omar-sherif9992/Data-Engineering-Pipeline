## to run the preprocessing docker image

- build the image```
sudo docker build -t preprocessing_image .

```

- run the image```
sudo docker run -it --rm -v $(pwd)/data:/app/data -v $(pwd)/src:/app/src --name preprocessing_container  preprocessing_image
```


## to run the whole pipeline through docker compose

```
sudo docker compose up --build --remove-orphans
```

## to check for the database in the terminal
    
- run the following command to check the volumes
```
sudo docker volume ls
```

- find the volume name of the postgres container
```
m3_49_3324_met_p1_10_2016_yellow_taxi_10_2016_postgres
```

- run another postgres container
```
sudo docker run -it --rm --name postgres_container -v m3_49_3324_met_p1_10_2016_yellow_taxi_10_2016_postgres:/var/lib/postgresql/data -e  POSTGRES_USER=root -e POSTGRES_PASSWORD=root  -e POSTGRES_DB=yellow_taxi -p 5432:5432 postgres:13
```

- open another terminal and run the following command to get into the postgres container
```
sudo docker exec -it postgres_container bash
```
or if docker compose was already running
```
sudo docker exec -it pgdatabase bash
```

- run the following command to get into the postgres database
```
 psql -d yellow_taxi
```

- run the following command to check the tables in the database
```
\dt
```



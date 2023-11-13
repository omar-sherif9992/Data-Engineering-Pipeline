## to run the preprocessing docker image

- build the image```
sudo docker build -t preprocessing_image .

```

- run the image
```
sudo docker run -it --rm -v $(pwd)/data:/app/data -v $(pwd)/src:/app/src --name preprocessing_container  preprocessing_image
```


## to run the whole pipeline through docker compose

```
sudo docker compose up --build --remove-orphans
```

## to check for the database in the terminal
    
```
sudo docker volume inspect green_taxi_10_2016_postgres
```

- run another postgres container
```
sudo docker run -it --rm --name postgres_container -v green_taxi_10_2016_postgres:/var/lib/postgresql/data postgres:13
```

- open another terminal and run the following command to get into the postgres container
```
sudo docker exec -it postgres_container bash
```

- run the following command to get into the postgres database
```
psql -U postgres
```

- run the following command to check the tables in the database
```
\dt
```



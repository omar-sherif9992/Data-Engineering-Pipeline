### to run the preprocessing docker image
- build the image```
sudo docker build -t preprocessing_image .
```
```bash
sudo docker run -it --rm -v $(pwd)/data:/app/data -v $(pwd)/src:/app/src --name preprocessing_container  preprocessing_image
```

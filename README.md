# DataScienceMLTest

The purpose of this project is to containerize the solution to
the machine learning test.

Build and run the image.
```
docker build  -t <image_name> .
docker run -it -v "${PWD}/DataScienceMLTest":"/home/worker/DataScienceMLTest" -p 8888:8888 -p 5001:5000 --name <container_name> <image_name>
```
### Ports description

| **Port** | **Description**                                                                         | **UI**                |
|----------|-----------------------------------------------------------------------------------------|-----------------------|
| 8888     | Jupyter lab<br/> (server needs to be <br/> launched manually, <br/>see **Start jupyter lab** <br/> section) | http://localhost:8888 |
| 5000     | Mlflow server                                                                           | http://localhost:5001 |


### Start interactive bash session
To start a interactive bash session, once the container is up and running, 
simply do
```
docker exec -it <container_name> bash
```

### Start jupyter lab
Once inside a container bash session,
```
jupyter lab --ip=0.0.0.0 --no-browser
```





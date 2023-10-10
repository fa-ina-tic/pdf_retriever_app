# project for retreiver model using pdf documents
settings for jupyter notebook settings

## image build
```shell
docker build --no-cache=true -t pdf_retriever:{version} .
```
run this code in dockerfile directory


## container run
```shell
docker run -it -p 8888:8888 -t pdf_retriever:{version} /bin/bash
```

## Access to jupyter notebook
```shell 
jupyter notebook --ip 0.0.0.0 --allow-root
```

# Dockerfile for training
FROM python:3.6


# Install dependencies
RUN apt-get update
RUN apt-get install -y python3-setuptools
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install Python library
RUN pip --no-cache-dir install \
    numpy==1.14.5 \
    pandas==0.23.1 \
    pandas-gbq==0.8.0 \
    gensim==3.4.0 \
    pyLDAvis==2.1.2 \
    google-cloud-storage==1.13.0 \
    google-api-python-client==1.7.4

WORKDIR /train
COPY train.py /train

ENTRYPOINT ["python", "train.py"]

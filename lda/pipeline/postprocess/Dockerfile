# Dockerfile for postprocessing
FROM python:3.6

# Install dependencies
RUN apt-get update
RUN apt-get install -y python3-setuptools
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install Python library
RUN pip --no-cache-dir install \
    pandas==0.23.1 \
    pandas-gbq==0.8.0 \
    google-cloud-storage==1.13.0

WORKDIR /postprocess
COPY postprocess.py /postprocess

ENTRYPOINT ["python", "postprocess.py"]

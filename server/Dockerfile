FROM python:3.7-slim-buster
COPY . /app
WORKDIR /app
RUN apt-get --quiet update --yes && apt-get --quiet upgrade --yes \
    && apt-get install -yqq libsndfile1 \
    && TMPDIR=/var/tmp pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "application.py"]
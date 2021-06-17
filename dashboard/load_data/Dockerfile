FROM python:3.6-slim 

WORKDIR /app 

COPY . /app

COPY ./data_db/* /app/

RUN pip install --trusted-host pypi.python.org -r requirements.txt

#CMD ["ls"]

CMD ["python", "load_data.py"]

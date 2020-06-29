#FROM tiangolo/uvicorn-gunicorn:python3.7-alpine3.8
FROM tiangolo/uvicorn-gunicorn:python3.7

RUN pip3 install --upgrade pip

WORKDIR /

RUN rm -rf /app
COPY ./app /app

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m nltk.downloader wordnet -d /app/data
RUN python3 -m nltk.downloader punkt -d /app/data

#COPY .env /.env
COPY gunicorn_conf.py /gunicorn_conf.py
COPY start.sh /start.sh
COPY start-reload.sh /start-reload.sh

EXPOSE 8050


#RUN cd ./app/core/test/ && export PYTHONPATH=../../../ && python3 Classifier.py 0
#RUN cd ./app/core/test/ && export PYTHONPATH=../../../ && python3 Classifier.py 1
#RUN cd ./app/core/test/ && export PYTHONPATH=../../../ && python3 Classifier.py 2
#RUN cd ./app/core/test/ && export PYTHONPATH=../../../ && python3 Classifier.py 3

#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 helpers.py

#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 train.py 0
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 train.py 1
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 train.py 2
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 train.py 3

#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 model_selection.py

#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 classify.py 0
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 classify.py 1
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 classify.py 2
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 classify.py 3

#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 feature_selection.py 0
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 feature_selection.py 1
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 feature_selection.py 2
#RUN cd ./app/test/ && export PYTHONPATH=../../ && python3 feature_selection.py 3

CMD /start.sh

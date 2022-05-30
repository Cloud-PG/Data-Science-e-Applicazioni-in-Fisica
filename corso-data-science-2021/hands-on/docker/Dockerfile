FROM jupyter/datascience-notebook:6b49f3337709

RUN pip3 install mlxtend numpy xgboost kaggle folium chardet fuzzywuzzy

RUN cd /tmp && git clone https://github.com/Kaggle/learntools.git && \
   pip3 install -e ./learntools/

RUN mkdir ~/.kaggle

COPY kaggle.json /home/jovyan/.kaggle/kaggle.json

USER root

RUN chown jovyan /home/jovyan/.kaggle/kaggle.json && chmod 600 /home/jovyan/.kaggle/kaggle.json

USER jovyan

RUN mkdir /home/jovyan/inputs

COPY ./03 /home/jovyan/inputs/03

COPY ./04 /home/jovyan/inputs/04

COPY ./05 /home/jovyan/inputs/05

USER root

RUN cd /home/jovyan/work && git clone https://github.com/Cloud-PG/Data-Science-e-Applicazioni-in-Fisica.git

RUN mv /home/jovyan/inputs/03 /home/jovyan/work/Data-Science-e-Applicazioni-in-Fisica/corso-data-science-2021/hands-on/03-data-cleaning/input \
 && mv /home/jovyan/inputs/04 /home/jovyan/work/Data-Science-e-Applicazioni-in-Fisica/corso-data-science-2021/hands-on/04-regression-and-clustering/input \
 && mv /home/jovyan/inputs/05 /home/jovyan/work/Data-Science-e-Applicazioni-in-Fisica/corso-data-science-2021/hands-on/05-geovis-and-dnn/input \
 && rm -fr /home/jovyan/inputs \
 && chown -R jovyan /home/jovyan

USER jovyan

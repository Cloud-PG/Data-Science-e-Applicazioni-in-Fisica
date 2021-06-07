#!/bin/bash 

apt install -y git && pip3 install mlxtend==0.18.0 numpy==1.16.2 xgboost kaggle folium

mkdir -p /workspace && mkdir -p /opt && \
   cd /opt && git clone https://github.com/Kaggle/learntools.git && \
   pip3 install -e ./learntools/

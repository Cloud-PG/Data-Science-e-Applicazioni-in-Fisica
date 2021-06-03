#!/bin/bash
set -ex

if [ -d input ]; then
    exit 0
fi
mkdir input

# Download the datasets used in the ML notebooks to correct relative_paths (../input/...)
DATASETS="dansbecker/melbourne-housing-snapshot dansbecker/aer-credit-card-data"

for slug in $DATASETS
do
    name=`echo $slug | cut -d '/' -f 2`
    dest="input/$name"
    mkdir -p $dest
    kaggle d download -p $dest --unzip $slug
done

COMPDATASETS="home-data-for-ml-course"

for comp in $COMPDATASETS
do
    dest="input/$comp"
    mkdir -p $dest
    kaggle competitions download $comp -p $dest
    cd $dest
    unzip ${comp}.zip
    chmod 700 *.csv
    cp *.csv ..
    cd ..
done

DATASETS="ryanholbrook/fe-course-data"

for slug in $DATASETS
do
    name=`echo $slug | cut -d '/' -f 2`
    dest="input/$name"
    mkdir -p $dest
    kaggle d download -p $dest --unzip $slug
done

DATASETS="dansbecker/melbourne-housing-snapshot dansbecker/aer-credit-card-data"

for slug in $DATASETS
do
    name=`echo $slug | cut -d '/' -f 2`
    dest="input/$name"
    mkdir -p $dest
    kaggle d download -p $dest --unzip $slug
done
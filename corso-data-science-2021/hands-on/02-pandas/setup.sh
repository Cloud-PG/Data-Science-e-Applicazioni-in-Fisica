#!/bin/bash
set -ex

if [ -d input ]; then
    exit 0
fi
mkdir input

DATASETS="zynicide/wine-reviews nolanbconaway/pitchfork-data dansbecker/powerlifting-database residentmario/things-on-reddit jpmiller/publicassistance datasnaek/youtube-new"

mkdir -p ~/.kaggle/

echo '{"username":"dciangot","key":"'${KAGGLE_KEY}'"}' >  ~/.kaggle/kaggle.json

for slug in $DATASETS
do
    name=`echo $slug | cut -d '/' -f 2`
    dest="input/$name"
    mkdir -p $dest
    kaggle d download -p $dest --unzip $slug
done

# Things on reddit is a zip of a zip. Kaggle d --unzip doesn't do a deep unzip, but kernels does
# unzip -q input/things-on-reddit/top-things.zip -d input/things-on-reddit/top-things

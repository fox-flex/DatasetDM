#!/bin/bash
set -e


eval "$(conda shell.bash hook)"
conda activate mgen

mkdir data && cd data
gdown 1AWy4Kwlj5l7cEOsQ1Y3KcUdXZfTvUIO2
gdown 1HTQU2l2uoTeA73p6J7Pvv_yzWnOvf54o
unzip UHRSD_TE.zip
unzip UHRSD_TR.zip
mkdir cap && cd cap
gdown 1sBP4z1XiXEenZTdJohUZGRZIljISHKAj
unzip cap.zip
mv UHRSD_TE/caption ../UHRSD_TE
mv UHRSD_TR/caption ../UHRSD_TR
cd ..
rm -rf cap
cd ..

ckpts_path="./dataset/ckpts/imagenet"
mkdir -p "$ckpts_path"
cd "$ckpts_path"
gdown 12lrOexljsyvFB30-ltbYXnIpQ8oP4lrW
unzip SD1.4.zip
cd -



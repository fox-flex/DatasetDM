#!/bin/bash
set -e


eval "$(conda shell.bash hook)"
conda activate base

mkdir data && cd data
gdown 1AWy4Kwlj5l7cEOsQ1Y3KcUdXZfTvUIO2
gdown 1HTQU2l2uoTeA73p6J7Pvv_yzWnOvf54o
unzip UHRSD_TE.zip
unzip UHRSD_TR.zip
mkdir cap && cd cap
gdown 1sBP4z1XiXEenZTdJohUZGRZIljISHKAj
unzip cap.zip
mv UHRSD_TE/caption ../UHRSD_TE
mv UHRSD_TE/caption ../UHRSD_TR
cd ..
rm -rf cap
cd ..

mkdir data/full && cd data
for name in caption image mask; do
    mkdir full/$name
    cp -r UHRSD_TE/$name/* full/$name
    cp -r UHRSD_TR/$name/* full/$name
done

ckpts_path="./dataset/ckpts/imagenet"
mkdir -p "$ckpts_path"
cd "$ckpts_path"
gdown 12lrOexljsyvFB30-ltbYXnIpQ8oP4lrW
unzip SD1.4.zip
cd -



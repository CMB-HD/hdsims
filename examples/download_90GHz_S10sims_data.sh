#!/bin/bash

datadir=${1:-$(pwd)}
echo "files will be placed in $datadir"

urls=(
  "https://lambda.gsfc.nasa.gov/data/tools/simulations/090_tsz_healpix.fits"
  "https://lambda.gsfc.nasa.gov/data/tools/simulations/090_ksz_healpix.fits"
  "https://lambda.gsfc.nasa.gov/data/tools/so_v3/healpix_4096_KappaeffLSStoCMBfullsky.fits"
  "https://lambda.gsfc.nasa.gov/data/tools/simulations/IRBasicPop.tar.gz"
  "https://lambda.gsfc.nasa.gov/data/tools/simulations/IRBlastPop.dat"
  "https://lambda.gsfc.nasa.gov/data/tools/simulations/radio.cat"
)

cd $datadir

for url in "${urls[@]}"; do
  echo "Downloading $url"
  wget -c "$url"
done

if [ -f "IRBasicPop.tar.gz" ]; then
  echo "Extracting IRBasicPop.tar.gz..."
  tar -xzf IRBasicPop.tar.gz
fi

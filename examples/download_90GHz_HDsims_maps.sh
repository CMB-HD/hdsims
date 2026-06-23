#!/bin/bash

datadir=${1:-$(pwd)}
echo "HD sims files will be placed in $datadir"

urls=(
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/cmbTQU0058_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/kappa_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/ksz_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/090_tsz_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/090_cib_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/090_radio_pixwin_11x11deg.fits"
)

cd $datadir

if [ ! -d "ra6dec6_10x10deg_hdsims" ]; then
  mkdir ra6dec6_10x10deg_hdsims
fi
cd ra6dec6_10x10deg_hdsims

for url in "${urls[@]}"; do
  echo "Downloading $url"
  wget -c "$url"
done

#!/bin/bash

datadir=${1:-$(pwd)}
echo "HD sims files will be placed in $datadir"

urls=(
  ""
)

cd $datadir

for url in "${urls[@]}"; do
  echo "Downloading $url"
  wget -c "$url"
done


#!/usr/bin/env bash

root -q -b makehist.C++ || true

for file in /data/schreihf/PvFinder/pv_201810* ; do
    file=$(basename -s .root $file)
    file=${file#pv_}
    echo "Processing: $file" 
    root -b -q "makehist.C+(\"$file\")" &
done
wait

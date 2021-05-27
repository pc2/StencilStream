#!/usr/bin/env bash

if [ ! -d rodinia_3.1 ]
then
    curl http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2 | tar -xjf -
fi

make -C rodinia_3.1/openmp/hotspot hotspot
cp rodinia_3.1/openmp/hotspot/hotspot hotspot_rodinia
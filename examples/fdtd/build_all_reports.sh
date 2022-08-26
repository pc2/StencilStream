#!/usr/bin/env bash

for mat in coef lut render
do
    for tdvs in inline device host
    do
        for exec in mono tiling
        do
            ./build.sh $mat $tdvs $exec report > "${mat}_${tdvs}_${exec}.log" &
        done
    done
done
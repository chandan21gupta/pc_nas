#!/bin/bash

set -x

runid=0
wd=1e-4

for dataset in nltcs
do
    python3 run.py --run=${runid} --dataset=${dataset}    --batch=500 --N=5 --modeltype=spn --lr=2e-2 --wd=${wd}
done

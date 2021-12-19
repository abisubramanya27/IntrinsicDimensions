#!/bin/bash

cp mount/abishek_xnli_new.py abishek_xnli_new.py
cp mount/abishek_xnli_zeroshot.py abishek_xnli_zeroshot.py

OUTPUT_DIR=/home/indic-analysis/container/checkpoints_mbert_xnli_sw/
DATASET="xnli"
CONFIG="sw"
# ID_arr=(0 100 500 10000 2000 5000 10000 12000 18000 20000 35000 50000 75000 100000 200000 500000)
ID_arr=(100 500 10000 2000 5000 10000 12000 18000)
declare -A ID_lr=( [0]=3e-5 [100]=1e-3 [500]=1e-3 [1000]=1e-3 [2000]=1e-3 [5000]=1e-3 [10000]=1e-3 [12000]=1e-3 [15000]=1e-3 [18000]=1e-3 [20000]=1e-3 [35000]=1e-3 [50000]=1e-3 [75000]=1e-3 [100000]=1e-3 [200000]=5e-4 [500000]=2e-4)

for ID in ${ID_arr[@]}; do
    python abishek_xnli_new.py
    python abishek_xnli_zeroshot.py
    if [ $ID != 0 ]; then
        rm -r $OUTPUT_DIR/bert-base-multilingual-cased_ID${ID}_lr${ID_lr[$ID]}_ml256
    fi
done

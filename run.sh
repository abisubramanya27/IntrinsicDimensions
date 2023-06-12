#!/bin/bash

cp mount/abishek_xnli_new.py abishek_xnli_new2.py
cp mount/abishek_xnli_zeroshot.py abishek_xnli_zeroshot2.py

export OUTPUT_DIR=/home/indic-analysis/container/checkpoints_mbert_xnli_fr
export DATASET="xnli"
export CONFIG="fr"
# ID_arr=(0 100 500 1000 2000 5000 10000 12000 18000 20000 35000 50000 75000 100000 200000 500000)
ID_arr=(0 20000 50000 75000 100000 200000) # 20000 35000 50000 75000 100000 200000)
# declare -A ID_lr=( [0]=3e-5 [100]=0.001 [500]=0.001 [1000]=0.001 [2000]=0.001 [5000]=0.001 [10000]=0.001 [12000]=0.001 [15000]=0.001 [18000]=0.001 [20000]=0.001 [35000]=0.001 [50000]=0.001 [75000]=0.001 [100000]=0.001 [200000]=0.0005 [500000]=0.0002)
declare -A ID_lr=( [0]=3e-5 [100]=0.001 [500]=0.001 [1000]=0.002 [2000]=0.002 [5000]=0.002 [10000]=0.002 [12000]=0.002 [15000]=0.002 [18000]=0.002 [20000]=0.002 [35000]=0.001 [50000]=0.001 [75000]=0.001 [100000]=0.0008 [200000]=0.0005 [500000]=0.0002)

for i in ${ID_arr[@]}; do
    export ID=$i
    python abishek_xnli_new2.py
    python abishek_xnli_zeroshot2.py
    if [ $i -ne 0 ]; then
        rm -r ${OUTPUT_DIR}/bert-base-multilingual-cased_ID${ID}_lr${ID_lr[$ID]}_ml256
    fi
done

#!/bin/bash


# CSV_TEST='./dataset_csv/test.csv'
# CSV_TEST='./dataset_csv/beijing_xiugao.csv'
# CSV_TEST='./dataset_csv/jiangsu_xiugao.csv'
# CSV_TEST='./dataset_csv/cptac_xiugao.csv'
# CSV_TEST='./dataset_csv/beijing2_xiugao.csv'
# CSV_TEST='./dataset_csv/diff_I.csv'
CSV_TEST='./dataset_csv/diff_III.csv'
EPOCH=300
DISTILL_TYPE='MaxMinS'
# echo 'Fold1...'
# CSV_TRAIN='./dataset_csv/fold1/train.csv'
# CSV_VAL='./dataset_csv/fold1/val.csv'

# python Main_DTFD_MIL.py \
#     --train_csv $CSV_TRAIN \
#     --val_csv $CSV_VAL \
#     --test_csv $CSV_TEST
echo 'Fold1...'
CSV_TRAIN='./dataset_csv/fold1/train.csv'
CSV_VAL='./dataset_csv/fold1/val.csv'
#LOG_DIR='./debug_log/beijing2/fold5'
LOG_DIR='./debug_log/diff_I/fold1'

python eval.py \
    --train_csv $CSV_TRAIN \
    --val_csv $CSV_VAL \
    --test_csv $CSV_TEST \
    --log_dir $LOG_DIR \
    --EPOCH $EPOCH \
    --distill_type $DISTILL_TYPE


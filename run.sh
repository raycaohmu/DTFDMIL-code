#!/bin/bash


CSV_TEST='./dataset_csv/test.csv'
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
LOG_DIR='./debug_log/fold1'

python Main_DTFD_MIL.py \
    --train_csv $CSV_TRAIN \
    --val_csv $CSV_VAL \
    --test_csv $CSV_TEST \
    --log_dir $LOG_DIR \
    --EPOCH $EPOCH \
    --distill_type $DISTILL_TYPE

echo 'Fold2...'
CSV_TRAIN='./dataset_csv/fold2/train.csv'
CSV_VAL='./dataset_csv/fold2/val.csv'
LOG_DIR='./debug_log/fold2'

python Main_DTFD_MIL.py \
    --train_csv $CSV_TRAIN \
    --val_csv $CSV_VAL \
    --test_csv $CSV_TEST \
    --log_dir $LOG_DIR \
    --EPOCH $EPOCH \
    --distill_type $DISTILL_TYPE

echo 'Fold3...'
CSV_TRAIN='./dataset_csv/fold3/train.csv'
CSV_VAL='./dataset_csv/fold3/val.csv'
LOG_DIR='./debug_log/fold3'

python Main_DTFD_MIL.py \
    --train_csv $CSV_TRAIN \
    --val_csv $CSV_VAL \
    --test_csv $CSV_TEST \
    --log_dir $LOG_DIR \
    --EPOCH $EPOCH \
    --distill_type $DISTILL_TYPE

echo 'Fold4...'
CSV_TRAIN='./dataset_csv/fold4/train.csv'
CSV_VAL='./dataset_csv/fold4/val.csv'
LOG_DIR='./debug_log/fold4'

python Main_DTFD_MIL.py \
    --train_csv $CSV_TRAIN \
    --val_csv $CSV_VAL \
    --test_csv $CSV_TEST \
    --log_dir $LOG_DIR \
    --EPOCH $EPOCH \
    --distill_type $DISTILL_TYPE

echo 'Fold5...'
CSV_TRAIN='./dataset_csv/fold5/train.csv'
CSV_VAL='./dataset_csv/fold5/val.csv'
LOG_DIR='./debug_log/fold5'

python Main_DTFD_MIL.py \
    --train_csv $CSV_TRAIN \
    --val_csv $CSV_VAL \
    --test_csv $CSV_TEST \
    --log_dir $LOG_DIR \
    --EPOCH $EPOCH \
    --distill_type $DISTILL_TYPE


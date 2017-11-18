#!/usr/bin/env bash

DATA_DIR=$1
TEST_OUT_FILENAME=$2
PEER_OUT_FILENAME=$3

cd src/
python predict.py $DATA_DIR testing_id.txt testing_data $TEST_OUT_FILENAME
python predict.py $DATA_DIR peer_review_id.txt peer_review $PEER_OUT_FILENAME

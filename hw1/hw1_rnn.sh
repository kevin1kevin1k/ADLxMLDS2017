#!/usr/bin/env bash

DATA_PATH=$1
CSV_FILENAME=$2

./download_models.sh

python model_rnn.py $DATA_PATH $CSV_FILENAME

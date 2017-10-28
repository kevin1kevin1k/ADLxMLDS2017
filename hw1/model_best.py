import sys

DATA_PATH = sys.argv[1]
CSV_FILENAME = sys.argv[2]

# Since this is ensemble, training code is not provided.

from predict import *

read_maps(DATA_PATH)
build_data_test(DATA_PATH)
y_predict = predict_ensemble('./models/')
write_csv(y_predict, CSV_FILENAME)

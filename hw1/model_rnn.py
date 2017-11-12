import sys

DATA_PATH = sys.argv[1]
CSV_FILENAME = sys.argv[2]

# Uncomment the following to enable training

'''
from train import *
build_data(DATA_PATH)
train_rnn('./models/')
'''

from predict import *

read_maps(DATA_PATH)
build_data_test(DATA_PATH)
y_predict = predict_single_model('./models/model_loss_015_0.0454_mapafter_lstm4_dp0.2_hidden500_adam0.0005.h5')
write_csv(y_predict, CSV_FILENAME)

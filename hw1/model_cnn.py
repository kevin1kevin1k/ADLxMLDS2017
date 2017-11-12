import sys

DATA_PATH = sys.argv[1]
CSV_FILENAME = sys.argv[2]

# Uncomment the following to enable training

'''
from train import *
build_data(DATA_PATH)
train_cnn_rnn('./models/')
'''

from predict import *

read_maps(DATA_PATH)
build_data_test(DATA_PATH)
y_predict = predict_single_model('./models/model_loss_020_0.1828_mapafter_conv1d_lstm2_dp0.2_hidden300_adam0.0005.h5')
write_csv(y_predict, CSV_FILENAME)

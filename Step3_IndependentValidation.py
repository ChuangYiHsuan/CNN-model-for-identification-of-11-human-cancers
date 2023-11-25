import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0" # assign specific graphic card (0 to 3)
import warnings
os.environ ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore warnings about version of python or tensorflow
warnings.filterwarnings('ignore') # "error", "ignore", "always", "default", "module" or "once"
import sys
from datetime import datetime as DT
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") # save files, not print on screen

import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential # build CNN model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization as B_nor
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
import keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

#MCC function
def MCC(y_true, y_pred): # Matthews correlation coefficient
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# 1. Load testing data and its label
# 2. Reshape input size 100*100
testing_set = sys.argv[1] # testing sample (one npy file containing all testing data)
test_sample = np.load(testing_set)
test_sample = test_sample.reshape((test_sample.shape[0], 100, 100, 1)) # sample size, height, width, 1 channel (because not RGB)

testing_label = sys.argv[2]
test_label = np.load(testing_label)
test_label_compare = test_label
test_label = np_utils.to_categorical(test_label)

filename_list = sys.argv[3]
test_sample_title = np.load(filename_list)

input_save_model = sys.argv[4] # the best training model
time = DT.now()
time = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute)
out=open(os.path.splitext(os.path.basename(testing_set))[0] + "_predicted_result_"+time,"w" )
print("Training model from "+input_save_model, file=out)
print("Indenpendent set from "+testing_set, file=out)

# If no previous *.save model, build a new CNN model
model = Sequential()
# Conv 1: frame 5*5, kernel number=64, input size 100*100
model.add(Conv2D(filters = 64, kernel_size = (5, 5), input_shape = (100, 100, 1)))
model.add(B_nor(axis = 2, epsilon = 1e-5)) # layer of batch normalization, normalize value to accelerate learning (accelerate convergence, avoid gradient vanishing/exploding )
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same")) # same: padding, "same" size of input
# Conv 2
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(B_nor(axis = 2, epsilon = 1e-5))
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
# Conv 3
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(B_nor(axis = 2, epsilon = 1e-5))
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
# Fully connect
model.add(Flatten())
# Hidden layers
model.add(Dense(1000, activation = "relu")) # 1st Hidden layer, 1000 neuron nodes, ReLU (Rectified Linear Unit)
model.add(Dense(600, activation = "relu")) # 2nd layer
model.add(Dense(80, activation = "relu")) # 3rd layer
model.add(Dense(12, activation = "softmax")) # 12 label classes, 11 cancers + 1 normal type

# load weight from previous training / best performance
try:
    model.load_weights(input_save_model)
    print("Successfully loading previous BEST training weights.")
except:
    print("Failed to load previous data, training new model below.")

# Executing model
optimizer_Adam = Adam(lr = 1e-4) # learning rate
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer_Adam, metrics = ['accuracy', MCC]) # metrics = ['accuracy', MCC], automaitcally call ACC, MCC function

# Independent Testing Results
scores = model.evaluate(test_sample, test_label, verbose = 1)
print("Independent test:\tAccuracy\t%.3f\tMCC\t%.3f\n" %( scores[1] , scores[2]), file=out)

# Use best trained model to do the prediction, and output confusion matrix
predict_x = model.predict(test_sample) 
classes_x = np.argmax(predict_x, axis=1)
num = 0
start = 1
print("Predict-result :", file=out)
for result in classes_x:
    label_from_title = str(test_sample_title[num]).split("-")
    if int(label_from_title[2]) != result:
        print("%i\tSample Name:\t%s\tActual:\t%i\tPredict:\t%i" %(start, test_sample_title[num], int(label_from_title[2]), result ), file=out)
        start += 1
    num += 1
# display confusion matrix
import pandas as pd
confusion_matrix = pd.crosstab(test_label_compare, classes_x, rownames=['Actual'], colnames=['Predicted'], margins=True)
print("\n\nConfusion Matrix:\n%s" %confusion_matrix)
print("\n\nConfusion Matrix:\n%s" %confusion_matrix, file=out)

out.close()

import os
import csv
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
image_directory = "Color Snippits"
master_file_path = "veg_or_unveg.csv"
X = []
y = []
i = 0
with open(master_file_path) as master_file:
    master_file_stream = csv.reader(master_file, delimiter=",")
    for row in master_file_stream:
        image = io.imread(image_directory + "/" + row[0])
        image = image / 255 #normalize pixel values (number between 0 and 1).
        X.append(image)
        if row[1] == "v":
            new_y = [1,0]
        else:
            new_y  = [0,1]
        y.append(new_y)
        i += 1
        
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_buckets = list()
y_buckets = list()
for bucket_number in range(0,100):
    X_bucket = list()
    y_bucket = list()
    for index in range (0,int(len(y_train)/100)):
        X_bucket.append(X_train[bucket_number * index + index])
        y_bucket.append(y_train[bucket_number * index + index])
    X_buckets.append(X_bucket)
    y_buckets.append(y_bucket)
    
    #experiments with fulll data set.

from keras.models import Sequential
from keras.layers import Conv2D, AvgPool2D, Activation, Flatten, Dense, MaxPool2D
from keras.optimizers import SGD
for interation in range(0,5):
    print("------------Iteration " + str(interation) + "---------------")
    for x in range(1,27):
        print("=====bucket number " + str(x) + "========")
        bucket_X_train = list()
        bucket_y_train = list()
        bucket_X_test = list()
        bucket_y_test = list()
        curve_X = list()
        curve_y = list()
        test_index = None
        if x <= 20:
            for index in range(0, x):
                bucket_X_train.extend(X_buckets[index])
                bucket_y_train.extend(y_buckets[index])
            test_index = x + 1
        else:
            multiple = x - 18
            final_index = multiple * 10
            for index in range(0,final_index):
                bucket_X_train.extend(X_buckets[index])
                bucket_y_train.extend(y_buckets[index])
            test_index = x + 1
        while test_index < 100:
            bucket_X_test.extend(X_buckets[test_index])
            bucket_y_test.extend(y_buckets[test_index])
            test_index += 1
        print(len(bucket_X_test))
        bucket_X_train = np.array(bucket_X_train)
        bucket_y_train = np.array(bucket_y_train)
        bucket_X_test = np.array(bucket_X_test)
        bucket_y_test = np.array(bucket_y_test)
        model = Sequential([
            Conv2D(input_shape=(31, 31, 3), filters=6, kernel_size=(4, 4), strides=(1, 1), padding='valid'),
            Activation('relu'),
            AvgPool2D(pool_size=(2, 2)),
            Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid'),
            Activation('relu'),
            AvgPool2D(pool_size=(2, 2)),
            Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), padding='valid'),
            Activation('relu'),
            Flatten(),
            Dense(84),
            Activation('relu'),
            Dense(2),
            Activation('softmax')
        ])
        optimizer = SGD(lr=10e-4)
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        model.fit(x=bucket_X_train,y=bucket_y_train,epochs=120,validation_data=(bucket_X_test, bucket_y_test),batch_size=1,verbose=1)

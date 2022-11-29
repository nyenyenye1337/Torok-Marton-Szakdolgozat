import sys
import os
from scipy.fftpack import fft
import numpy as np
import csv
import keras
from keras import utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from vis.visualization import visualize_saliency
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.set_printoptions(threshold=sys.maxsize)

# Initializing code

with_gyri = False
with_bmap = False

print('Choose the trainng subject (0-15): ')
input1 = input()
train_subject = int(input1)

print('Do you want to work with the Gyri data? (Y/N)')
input2 = input()
if str(input2) == 'Y' or str(input2) == 'y':
    with_gyri = True

print('Do you want to work with the BMAP data? (Y/N)')
input4 = input()
if str(input4) == 'Y' or str(input4) == 'y':
    with_bmap = True


# Read Gyri data

if with_gyri:
    gyridata = np.ndarray((16, 12), dtype=np.int)
    with open("Gyridata.csv", newline='') as gyrifile_csv:
        r = csv.reader(gyrifile_csv, delimiter=',')
        i_row = 0
        for row in r:
            for i_col in range(gyridata.shape[1]):
                gyridata[i_row, i_col] = row[i_col]
            i_row = i_row + 1


    gyri_types = ['Superior Frontal', 'Middle Frontal', 'Inferior Frontal', 'Superior Temporal', 'Middle Temporal',
                  'Inferior Temporal', 'Precentral', 'Postcentral', 'Supramarginal', 'Cingulate', 'Angular',
                  'Superior Parietal']

    gyri_df = pd.DataFrame(gyridata, columns=gyri_types)

    # Getting the best match for the subject

    train_row = gyridata[train_subject, :]
    choosing_array = gyridata.copy()
    choosing_array[train_subject, :] = choosing_array[train_subject, :] + 10
    choosing_array = np.abs(np.sum(choosing_array - train_row, axis=1))
    best_match = np.min(choosing_array)
    test_subject_array = np.where(choosing_array == best_match)[0]

    if int(test_subject_array.shape[0]) > 1:
        print('Best matches for subject S' + str(train_subject).zfill(2) + ' are:')
        num = 0
        while num < test_subject_array.shape[0]:
            print('S' + str(test_subject_array[num]).zfill(2))
            num += 1
        print('Choose the subject you want to proceed with (just the number): ')
        input3 = input()
        test_subject = input3
    if int(test_subject_array.shape[0]) == 1:
        print('Best match for subject S' + str(train_subject).zfill(2) + ' is subject S' + str(
            test_subject_array[0]).zfill(
            2))
        test_subject = test_subject_array[0]
    else:
        test_subject = train_subject

    # Display the test subject and training subject

    print('Training subject is S' + str(train_subject).zfill(2) + ' and the testing subject is S' + str(
        test_subject).zfill(
        2))
    print(
        'The following table shows us which gyri are shared between the two subjects (the subject numbers are row and gyri are columns, there is a match where both the values are 1):')
    print(gyri_df.loc[[train_subject, test_subject]])
else:
    test_subject = train_subject

# Read BMAP data

if with_bmap:
    train_bmapdata = np.ndarray((16, 4), dtype=np.int)
    ifn_config = "S" + str(train_subject).zfill(2) + "/S" + str(train_subject).zfill(2) + "_BMAP.csv"
    with open(ifn_config, newline='') as train_bmap_csv:
        r = csv.reader(train_bmap_csv, delimiter=',')
        i_row = 0
        for row in r:
            for i_col in range(train_bmapdata.shape[1]):
                train_bmapdata[i_row, i_col] = row[i_col]
            i_row = i_row + 1
    print("Train subject BMAP data:")
    print(train_bmapdata.transpose())

    if not train_subject == test_subject:
        test_bmapdata = np.ndarray((16, 4), dtype=np.int)
        ifn_config = "S" + str(test_subject).zfill(2) + "/S" + str(test_subject).zfill(2) + "_BMAP.csv"
        with open(ifn_config, newline='') as test_bmap_csv:
            r = csv.reader(test_bmap_csv, delimiter=',')
            i_row = 0
            for row in r:
                for i_col in range(test_bmapdata.shape[1]):
                    test_bmapdata[i_row, i_col] = row[i_col]
                i_row = i_row + 1
        print("Test subject BMAP data:")
        print(test_bmapdata.transpose())

# Read trial timestamps
# Training subject

train_filedata = np.ndarray((16, 5), dtype=np.int)
with open("Filedata.csv", newline='') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    i_row = 0
    for row in r:
        for i_col in range(train_filedata.shape[1]):
            train_filedata[i_row, i_col] = row[i_col]
        i_row = i_row + 1

ifn_config = "Configs/S" + str(train_subject).zfill(2) + ".csv"
train_config = np.ndarray((train_filedata[train_subject, 1], 5), dtype=np.int)
with open(ifn_config, newline='') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    i_row = 0
    for row in r:
        for i_col in range(train_config.shape[1]):
            train_config[i_row, i_col] = row[i_col]
        i_row = i_row + 1

# Test subject
if not train_subject == test_subject:
    test_filedata = np.ndarray((16, 5), dtype=np.int)
    with open("Filedata.csv", newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        i_row = 0
        for row in r:
            for i_col in range(test_filedata.shape[1]):
                test_filedata[i_row, i_col] = row[i_col]
            i_row = i_row + 1

    ifn_config = "Configs/S" + str(test_subject).zfill(2) + ".csv"
    test_config = np.ndarray((test_filedata[test_subject, 1], 5), dtype=np.int)
    with open(ifn_config, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        i_row = 0
        for row in r:
            for i_col in range(test_config.shape[1]):
                test_config[i_row, i_col] = row[i_col]
            i_row = i_row + 1

# Read ECoG/EEG data
# Train subject

window_length = int(train_filedata[train_subject, 4] / 1)
offset = int(train_filedata[train_subject, 1] / 1)
n_chunk = 9
i_length = 39
train_bigdata = np.zeros(
    (int(train_filedata[train_subject, 0] * n_chunk * i_length), window_length, train_filedata[train_subject, 1]),
    dtype=float)
label = np.zeros((int(train_filedata[train_subject, 0] * n_chunk * i_length),), dtype=int)

for i_file in range(train_filedata[train_subject, 0]):

    ifn = "S" + str(train_subject).zfill(2) + "/S" + str(train_subject).zfill(2) + "R" + str(i_file)
    ifn_csv = ifn + ".csv"
    ifn_dat = ifn + ".dat"

    with open(ifn_csv, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        n_rows = 0
        for row in r:
            n_rows = n_rows + 1

    indices = np.ndarray((n_rows,), dtype=np.int)
    with open(ifn_csv, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        i_row = 0
        for row in r:
            indices[i_row] = row[0]
            i_row = i_row + 1

    filelength = int(
        os.path.getsize(ifn_dat) / (train_filedata[train_subject, 1] * (train_filedata[train_subject, 2] / 8)))
    indices_to_plot = np.zeros((filelength,))
    indices_to_plot[indices] = 1

    gates = np.zeros((filelength,))
    last_value = 0
    for i in range(filelength):
        gates[i] = last_value
        if indices_to_plot[i] == 1:
            if last_value == 0:
                last_value = 1
            else:
                last_value = 0
        gates[i] = last_value

    datatype = np.uint16
    if train_filedata[train_subject, 2] == 16 and train_filedata[train_subject, 3] == 1:
        datatype = np.int16
    if train_filedata[train_subject, 2] == 32 and train_filedata[train_subject, 3] == 0:
        datatype = np.uint32
    if train_filedata[train_subject, 2] == 32 and train_filedata[train_subject, 3] == 1:
        datatype = np.int32

    data = np.fromfile(ifn_dat, dtype=datatype)
    data = data.reshape((filelength, train_filedata[train_subject, 1])).astype(float)

    for i in range(i_length):
        for j in range(n_chunk):
            data_index = i_file * i_length * n_chunk + i * n_chunk + j
            train_bigdata[data_index, :, :] = data[indices[i] + (j + 1) * offset: indices[i] + (
                        j + 1) * offset + window_length, :].reshape(1, window_length, train_filedata[train_subject, 1])
            label[data_index] = gates[indices[i] + 10] * (i_file + 1)

# Test subject

if not train_subject == test_subject:
    window_length = int(test_filedata[test_subject, 4] / 1)
    offset = int(test_filedata[test_subject, 1] / 1)
    n_chunk = 9
    i_length = 39
    test_bigdata = np.zeros(
        (int(test_filedata[test_subject, 0] * n_chunk * i_length), window_length, test_filedata[test_subject, 1]),
        dtype=float)
    label = np.zeros((int(test_filedata[test_subject, 0] * n_chunk * i_length),), dtype=int)

    for i_file in range(test_filedata[test_subject, 0]):

        ifn = "S" + str(test_subject).zfill(2) + "/S" + str(test_subject).zfill(2) + "R" + str(i_file)
        ifn_csv = ifn + ".csv"
        ifn_dat = ifn + ".dat"

        with open(ifn_csv, newline='') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            n_rows = 0
            for row in r:
                n_rows = n_rows + 1

        indices = np.ndarray((n_rows,), dtype=np.int)
        with open(ifn_csv, newline='') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            i_row = 0
            for row in r:
                indices[i_row] = row[0]
                i_row = i_row + 1

        filelength = int(
            os.path.getsize(ifn_dat) / (test_filedata[test_subject, 1] * (test_filedata[test_subject, 2] / 8)))
        indices_to_plot = np.zeros((filelength,))
        indices_to_plot[indices] = 1

        gates = np.zeros((filelength,))
        last_value = 0
        for i in range(filelength):
            gates[i] = last_value
            if indices_to_plot[i] == 1:
                if last_value == 0:
                    last_value = 1
                else:
                    last_value = 0
            gates[i] = last_value

        datatype = np.uint16
        if test_filedata[test_subject, 2] == 16 and test_filedata[test_subject, 3] == 1:
            datatype = np.int16
        if test_filedata[test_subject, 2] == 32 and test_filedata[test_subject, 3] == 0:
            datatype = np.uint32
        if test_filedata[test_subject, 2] == 32 and test_filedata[test_subject, 3] == 1:
            datatype = np.int32

        data = np.fromfile(ifn_dat, dtype=datatype)
        data = data.reshape((filelength, test_filedata[test_subject, 1])).astype(float)

        for i in range(i_length):
            for j in range(n_chunk):
                data_index = i_file * i_length * n_chunk + i * n_chunk + j
                test_bigdata[data_index, :, :] = data[indices[i] + (j + 1) * offset: indices[i] + (
                        j + 1) * offset + window_length, :].reshape(1, window_length, test_filedata[test_subject, 1])
                label[data_index] = gates[indices[i] + 10] * (i_file + 1)

# # Data preprocessing

# Delete channels contaminated with artefacts and EEG channels
# notfeasible = np.where(train_config[:, 2] != 0)[0]
# train_bigdata = np.delete(train_bigdata, notfeasible, axis=2)
train_bigdata = np.abs(fft(train_bigdata, axis=1))

if not train_subject == test_subject:
    # notfeasible = np.where(test_config[:, 2] != 0)[0]
    # test_bigdata = np.delete(test_bigdata, notfeasible, axis=2)
    test_bigdata = np.abs(fft(test_bigdata, axis=1))


# Filter 50 Hz harmonics
train_bigdata[:, int((window_length * 50 / train_filedata[train_subject, 4])), :] = 0
train_bigdata[:, int((window_length * 100 / train_filedata[train_subject, 4])), :] = 0
train_bigdata[:, int((window_length * 150 / train_filedata[train_subject, 4])), :] = 0

if not train_subject == test_subject:
    test_bigdata[:, int((window_length * 50 / test_filedata[test_subject, 4])), :] = 0
    test_bigdata[:, int((window_length * 100 / test_filedata[test_subject, 4])), :] = 0
    test_bigdata[:, int((window_length * 150 / test_filedata[test_subject, 4])), :] = 0

# Restrict data to the (0, 200] Hz range
train_bigdata = np.expand_dims(train_bigdata[:, 1:int((window_length * 200 / train_filedata[train_subject, 4])), :],
                               axis=3)

if not train_subject == test_subject:
    test_bigdata = np.expand_dims(test_bigdata[:, 1:int((window_length * 200 / test_filedata[test_subject, 4])), :],
                                  axis=3)

# Standardize data
train_bigdata = (train_bigdata - np.mean(train_bigdata)) / (np.std(train_bigdata))

if not train_subject == test_subject:
    test_bigdata = (test_bigdata - np.mean(test_bigdata)) / (np.std(test_bigdata))

data_to_draw = train_bigdata.reshape(
    (n_chunk, i_length, train_filedata[train_subject, 0], train_bigdata.shape[1], train_bigdata.shape[2]))
active_indices = 2 * np.arange(int(np.floor(i_length / 2)))
active_avg = np.average(data_to_draw[:, active_indices, :, :, :], axis=1)
passive_avg = np.squeeze(np.average(np.average(data_to_draw[:, active_indices + 1, :, :, :], axis=1), axis=1))
print(active_avg.shape)

shuffle_indices = np.arange(train_bigdata.shape[0])
np.random.shuffle(shuffle_indices)
train_bigdata = train_bigdata[shuffle_indices, :, :]
label = label[shuffle_indices]
label_cat = keras.utils.to_categorical(label, num_classes=train_filedata[train_subject, 0] + 1)
training_rate = 0.8



# FFT comparison WIP
# print('plotting')
# plt.plot(train_bigdata.shape[0], test_bigdata.shape[0])
# plt.show()
# print('done')

# 2D convnet
input_layer = keras.Input(shape=train_bigdata[0].shape, name="input_layer")
c1 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(input_layer)
do1 = keras.layers.Dropout(0.5)(c1)
bn1 = keras.layers.BatchNormalization()(do1)
c2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(bn1)
do2 = keras.layers.Dropout(0.5)(c2)
bn2 = keras.layers.BatchNormalization()(do2)
c3 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(bn2)
do3 = keras.layers.Dropout(0.5)(c3)
bn3 = keras.layers.BatchNormalization()(do3)
c4 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(bn3)
do4 = keras.layers.Dropout(0.5)(c4)
bn4 = keras.layers.BatchNormalization()(do4)

flatten = keras.layers.Flatten()(bn4)
dense1 = keras.layers.Dense(128, activation="relu")(flatten)
do0 = keras.layers.Dropout(0.5)(dense1)
bn0 = keras.layers.BatchNormalization()(do0)

out = keras.layers.Dense(label_cat.shape[1], activation="softmax", name="out")(bn0)
model = keras.Model(inputs=[input_layer], outputs=[out])
optimizer = keras.optimizers.RMSprop(lr=0.1, decay=0.001)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10),
             keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_acc', save_best_only=True)]

# Training and test databases
train_input = train_bigdata[0:int(train_bigdata.shape[0] * training_rate), :, :]

if not train_subject == test_subject:
    test_input = test_bigdata[int(test_bigdata.shape[0] * training_rate):test_bigdata.shape[0], :, :]

# Class weights for balanced training
cw = np.sum(label_cat[0:int(train_bigdata.shape[0] * training_rate), :], axis=0)
cw = cw[0] / cw

# Convnet training
model.fit(train_input, {"out": label_cat[0:int(train_bigdata.shape[0] * training_rate), :]}, callbacks=callbacks,
          epochs=50, verbose=2,
          validation_split=0.25, shuffle=True, class_weight=cw, batch_size=5)

# We evaluate the best model
model = keras.models.load_model('best_model.h5')

# Prediction
if train_subject == test_subject:
    label_predicted = model.predict(train_input)
else:
    label_predicted = model.predict(test_input)

label_predicted0 = np.zeros(train_bigdata.shape[0] - int(train_bigdata.shape[0] * training_rate))

# Confusion matrix
for i in range(train_bigdata.shape[0] - int(train_bigdata.shape[0] * training_rate)):
    label_predicted0[i] = np.argmax(label_predicted[i, :])
cm = confusion_matrix(label[int(train_bigdata.shape[0] * training_rate):train_bigdata.shape[0]], label_predicted0)
print(cm)

# Evaluate results
# train_bigdata-t test_bigdata-ra cserélés vagy fordítva (eredeti train_bigdata)
test_label = label_cat[int(train_bigdata.shape[0] * training_rate):train_bigdata.shape[0]]
score = model.evaluate(train_input, {"out": test_label})
print(score)

########################################################################################################################


# # Produce saliency maps from test samples
# test_reconstructed = np.zeros((test_input.shape[0], test_input.shape[1], test_input.shape[2]))
# for i_sample in range(test_label.shape[0]):
#     class_id = np.argmax(test_label[i_sample, :])
#     print(class_id)
#     test_reconstructed[i_sample, :, :] = visualize_saliency(model=model, layer_idx=-1, filter_indices=class_id, seed_input=test_input[i_sample, :, :])
#
# # Fit a small dense network to the best features
# n_greatest_vec = [1, 10, 100, 1000]
# for n_greatest in n_greatest_vec:
#     sal_imp = np.zeros((test_input.shape[1], test_input.shape[2]))
#
#     # Best features of each category
#     for i in range(test_label.shape[1]):
#         x = np.where(test_label[:, i] == 1)[0]
#         y0 = np.squeeze(np.average(test_input[x, :, :], axis=0))
#         y = np.squeeze(np.average(test_reconstructed[x, :, :], axis=0))
#
#         greatest = np.sort(y, axis=None)
#         greatest = greatest[greatest.shape[0] - n_greatest:greatest.shape[0]]
#
#         for val in greatest:
#             sal_imp = sal_imp + np.where(y == val, 1, 0)
#
#         fig, ax = plt.subplots()
#         im = ax.imshow(y)
#         cbar = ax.figure.colorbar(im, ax=ax)
#         plt.show()
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(sal_imp)
#     cbar = ax.figure.colorbar(im, ax=ax)
#     plt.show()
#
#     # Extract the best features from each input sample, create train and test databases
#     sal_imp = sal_imp.reshape((sal_imp.shape[0]*sal_imp.shape[1]))
#     mask = np.where(sal_imp > 0)
#     train_pruned = train_input.reshape((train_input.shape[0], train_input.shape[1]*train_input.shape[2]))
#     train_pruned = train_pruned[:, mask]
#     test_pruned = test_input.reshape((test_input.shape[0], test_input.shape[1]*test_input.shape[2]))
#     test_pruned = test_pruned[:, mask]
#
#     # Description of dense layer
#     m2_input_layer = keras.Input(shape=train_pruned[0].shape, name="m2_input_layer")
#     m2_flatten = keras.layers.Flatten()(m2_input_layer)
#     m2_dense0 = keras.layers.Dense(256, activation="relu")(m2_flatten)
#     m2_do0 = keras.layers.Dropout(0.5)(m2_dense0)
#     m2_bn0 = keras.layers.BatchNormalization()(m2_do0)
#     m2_out = keras.layers.Dense(label_cat.shape[1], activation="softmax", name="m2_out")(m2_bn0)
#
#     model2 = keras.Model(inputs=[m2_input_layer], outputs=[m2_out])
#     model2.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
#     callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10),
#                  keras.callbacks.ModelCheckpoint(filepath='best_model2.h5', monitor='val_acc', save_best_only=True)]
#
#     # Train model
#     model2.fit(train_pruned, {"m2_out": label_cat[0:int(train_bigdata.shape[0]*training_rate), :]}, callbacks=callbacks, epochs=50, verbose=2,
#               validation_split=0.25, shuffle=True, class_weight=cw, batch_size=5)
#
#     # We use the best model for prediction
#     model2 = keras.models.load_model('best_model2.h5')
#
#     # Prediction and evaluation
#     label_predicted = model2.predict(test_pruned)
#     label_predicted0 = np.zeros(train_bigdata.shape[0] - int(train_bigdata.shape[0] * training_rate))
#     for i in range(train_bigdata.shape[0] - int(train_bigdata.shape[0] * training_rate)):
#         label_predicted0[i] = np.argmax(label_predicted[i, :])
#     cm = confusion_matrix(label[int(train_bigdata.shape[0]*training_rate):train_bigdata.shape[0]], label_predicted0)
#     print(cm)
#     test_label = label_cat[int(train_bigdata.shape[0]*training_rate):train_bigdata.shape[0]]
#     score = model2.evaluate(test_pruned, {"m2_out": test_label})
#     print(score)

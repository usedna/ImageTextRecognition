import datetime
import json

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from OCR.utils import pre_processing, to_categorical


############################
all_path_record = "../raw_data/record.json"
cv2_path_record = "../raw_data/record_cv2.json"
pil_path_record = "../raw_data/record_pil.json"

earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)
mc = tf.keras.callbacks.ModelCheckpoint('OCR/models_saved/ocr_model_digits1.h5',
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True)
############################

def load_data_record(record_path: str):
    with open(record_path, 'r') as jf:
        data_loaded = json.load(jf)
    return data_loaded

def prepare_data(images_paths: dict,
                 image_dimensions: tuple = (28, 28, 3)):


    no_of_classes = 0

    images = []
    classes = []

    print("Total number of classes: ", no_of_classes)
    print("Importing ...")

    for img_class, paths, in images_paths.items():
        if not img_class.isalnum():
            continue

        for path in tqdm(paths[:6328]):
            curImg = cv2.imread(path)

            curImg = 255 - curImg
            curImg[curImg < 100] = 0
            curImg[curImg > 100] = 255

            curImg = cv2.resize(curImg, (image_dimensions[0], image_dimensions[1]))
            images.append(curImg)
            classes.append(img_class)

    no_of_classes = len(set(classes))

    print("Total number of classes: ", no_of_classes)
    print("Data size: ", len(images))

    images = np.array(images)
    classes = np.array(classes)

    print(images.shape)
    print(classes.shape)

    return images, classes, no_of_classes


def prepare_training_data(images: np.ndarray,
                          classes: np.ndarray,
                          test_ratio: float = 0.2,
                          validation_ratio: float = 0.2):

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(images, classes, test_size=test_ratio)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)
    print(X_train.shape)
    print(X_test.shape)
    print(X_validation.shape)

    X_train = np.array(list(map(pre_processing, X_train)))
    X_test = np.array(list(map(pre_processing, X_test)))
    X_validation = np.array(list(map(pre_processing, X_validation)))

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_validation = to_categorical(y_validation)

    return (X_train, y_train), (X_test, y_test), (X_validation, y_validation)


def create_model(no_of_classes: int,
                 input_shape: tuple = (28, 28, 3),
                 no_of_filters: int = 64,
                 size_of_filter1: tuple = (5, 5),
                 size_of_filter2: tuple = (3, 3),
                 size_of_pool: tuple = (2, 2),
                 no_of_nodes: int = 500,
                 dropout_rate: int = 0.25,
                 learning_rate=0.001,
                 dropout=False,
                 device: str = "GPU:0"):


    with tf.device(device):
        model = tf.keras.Sequential()

        model.add((tf.keras.layers.Conv2D(no_of_filters,
                          size_of_filter1,
                          input_shape=(input_shape[0], input_shape[1], 1),
                          activation='relu')))
        model.add((tf.keras.layers.Conv2D(no_of_filters,
                          size_of_filter1,
                          activation='relu')))
        model.add((tf.keras.layers.Conv2D(no_of_filters,
                                          size_of_filter1,
                                          activation='relu')))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=size_of_pool))

        model.add((tf.keras.layers.Conv2D(no_of_filters // 2,
                          size_of_filter2,
                          activation='relu')))
        model.add((tf.keras.layers.Conv2D(no_of_filters // 2,
                          size_of_filter2,
                          activation='relu')))
        model.add((tf.keras.layers.Conv2D(no_of_filters // 2,
                                          size_of_filter2,
                                          activation='relu')))


        model.add(tf.keras.layers.MaxPooling2D(pool_size=size_of_pool))

        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(no_of_nodes, activation='relu'))

        if dropout:
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(no_of_classes, activation='softmax'))

        model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


def train_model(model: tf.keras.Sequential,
                train_data: tuple,
                validation_data: tuple,
                batch_size: int = 32,
                epoch: int = 10,
                call_backs = None):

    X_train, y_train = train_data
    X_validation, y_validation = validation_data

    steps_per_epoch = len(X_train) // batch_size

    dataGen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)

    dataGen.fit(X_train)

    history = model.fit(dataGen.flow(X_train, y_train,
                                     batch_size=batch_size),
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=epoch,
                                     validation_data=(X_validation, y_validation),
                                     shuffle=1,
                                     callbacks=call_backs)

    return model, history


def model_eval(model, test_data, verbose = 2, call_backs = None):
    score = model.evaluate(test_data[0], test_data[1], verbose=verbose, callbacks=call_backs)
    print("Test Loss = ", score[0])
    print("Test Accuracy = ", score[1])

    return score


def get_tensorboard_callback(log_file_name):
    log_dir = "logs/" + log_file_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return tensorboard_callback


def start_train(img, clss, num_of_classes, model=None, model_path=None, batch_size=32, epoch=100):

    if model is not None:
        ocr_model = model

    elif model_path is not None:
        ocr_model = tf.keras.models.load_model(model_path)

    else:
        ocr_model = create_model(num_of_classes, dropout=False)

    print(ocr_model.summary())

    train_split, test_split, val_split = prepare_training_data(img, clss)

    ocr_model, history = train_model(ocr_model,
                                     train_split,
                                     val_split,
                                     epoch=epoch,
                                     batch_size=batch_size,
                                     call_backs=[get_tensorboard_callback('fit3')])

    model_eval(ocr_model, test_split, call_backs=[get_tensorboard_callback('eval3')])

    return ocr_model

data = load_data_record(all_path_record)

img, clss, num_of_classes = prepare_data(data)

for i in range(10, 11):
    bs = 2**i
    first = start_train(img, clss, num_of_classes, batch_size=bs, epoch=500)

    tf.keras.models.save_model(first, f'OCR/models_saved/v500bs{bs}/ocr_model_500e.hdf5')

import os
import numpy as np
import argparse

import tensorflow as tf
import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras import backend as K

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def video_base_model():
    
    _MASKING = 0
    _MAX_LEN = 6

    model = keras.models.Sequential()
    model.add(Masking(mask_value=_MASKING, input_shape=(_MAX_LEN, 4096)))
    model.add((LSTM(128, activation='relu')))
    model.add(Dense(64, activation='relu',name='BottleNeck'))
    model.add(Dense(7, name='emotion',activation='softmax'))
    model.summary()
    return model

def main():

    # Load data and label
    _ROOT_PATH = 'features/'

    x_train = np.load(_ROOT_PATH + "video_train.npy")
    x_val = np.load(_ROOT_PATH + "video_val.npy")
    y_train = np.load(_ROOT_PATH + "label_train.npy")
    y_val = np.load(_ROOT_PATH + "label_val.npy")

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=7)
    y_val = keras.utils.to_categorical(y_val, num_classes=7)

    # Training Parameter setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 30},
            gpu_options=gpu_options)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
 
    # Model build
    model = video_base_model()

    model_path = 'model/' + 'video_model_' + 'acc_{val_acc:.4f}.h5'    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
            verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0005,
            patience=30, verbose=1, mode='auto')
 
    # Train
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
            metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=300,
            validation_data=(x_val,y_val), verbose=1,
            callbacks=[early_stopping, checkpoint])
    
    # Evaluation
    score = model.evaluate(x_val, y_val, batch_size=256)

    print(score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    K.clear_session()

if __name__ == '__main__':
    main()

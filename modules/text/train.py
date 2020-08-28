import os
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
import numpy as np
import glob

def text_base_model():
    _MAX_LEN = 30
    model = keras.models.Sequential()
    model.add(Masking(mask_value=0, input_shape=(_MAX_LEN, 200)))
    model.add(Bidirectional(LSTM(128, activation='relu')))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', name='BottleNeck'))
    model.add(Dropout(0.5))
    model.add(Dense(7, name='Emotion',activation='softmax'))
    model.summary()
    return model

def main():

    # Load Training & Validation data 
    _ROOT_PATH = "dataset/"
    x_train = np.load(_ROOT_PATH + "text_train.npy")
    x_val = np.load(_ROOT_PATH + "text_val.npy")
    y_train = np.load(_ROOT_PATH + "label_train.npy")
    y_val = np.load(_ROOT_PATH + "label_val.npy")

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes = 7)
    y_val = keras.utils.to_categorical(y_val, num_classes = 7)

    # Training Parameter setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 30},gpu_options=gpu_options)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    # Model build
    model = text_base_model() 
    
    # Model Check point
    model_path = 'model/' + 'text_model_' + 'acc_{val_acc:.4f}.h5'    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta = 0.0005, patience = 30, verbose = 1, mode='auto')
    
    # Training
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size = 512, epochs = 256, validation_data= (x_val,y_val), verbose=1, callbacks=[early_stopping, checkpoint])

    ### Evaluation
    score = model.evaluate(x_val, y_val, batch_size = 256)

    print(score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()

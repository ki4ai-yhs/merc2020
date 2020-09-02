import os
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Model
from keras.layers import *
import numpy as np
import glob
from keras import backend as K

def multimodal_base_model():
    
    t_input = Input(shape=(32,))
    t = t_input
    t = Dense(16, name="text_inter", activation='relu')(t)

    v_input = Input(shape=(64,))
    v = v_input
    v = Dense(16, name="video_inter", activation='relu')(v)
    
    s_input = Input(shape=(64,))
    s = s_input
    s = Dense(16, name="speech_inter", activation='relu')(s)

    x = concatenate([t,v,s])

    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(7, name='emotion',activation='softmax')(x)
    model = Model(inputs=[t_input,v_input,s_input], outputs=output, name="multimodal_model")

    model.summary()
    return model

def main():

    # Load Training & Validation data 
    _ROOT_PATH = "features/"
    t_train = np.load(_ROOT_PATH + "text_BN_train.npy")
    t_val = np.load(_ROOT_PATH + "text_BN_val.npy")
    v_train = np.load(_ROOT_PATH + "video_BN_train.npy")
    v_val = np.load(_ROOT_PATH + "video_BN_val.npy")
    s_train = np.load(_ROOT_PATH + "speech_BN_train.npy")
    s_val = np.load(_ROOT_PATH + "speech_BN_val.npy")
    y_train = np.load(_ROOT_PATH + "label_train.npy")
    y_val = np.load(_ROOT_PATH + "label_val.npy")


    x_train = [t_train, v_train, s_train]
    x_val = [t_val, v_val, s_val]

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes = 7)
    y_val = keras.utils.to_categorical(y_val, num_classes = 7)

    # Training Parameter setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 30},gpu_options=gpu_options)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    # Model build
    model = multimodal_base_model() 
    
    # Model Check point
    model_path = 'model/' + 'multimodal_model_' + 'acc_{val_acc:.4f}.h5'    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta = 0.0005, patience = 30, verbose = 1, mode='auto')
    
    # Training
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size = 128, epochs = 256, validation_data= (x_val,y_val), verbose=1, callbacks=[early_stopping, checkpoint])

    ### Evaluation
    score = model.evaluate(x_val, y_val, batch_size = 256)

    print(score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    K.clear_session()

if __name__ == '__main__':
    main()

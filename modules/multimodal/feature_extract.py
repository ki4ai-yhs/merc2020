## will be deleted
import os
import utils
os.environ["CUDA_VISIBLE_DEVICES"]=str(utils.pick_gpu_lowest_memory())

import tensorflow as tf
import keras
import numpy as np
import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import Model
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

textModelPath = 'inputs/text_model_acc_0.4523.h5'
videoModelPath = 'inputs/video_model_acc_0.2971.h5'

def inter_model_load(model_path):
    text_model=load_model(model_path)
    inter_layer_model = Model(inputs=text_model.input, outputs=text_model.get_layer('BottleNeck').output)
    return inter_layer_model

def main(modal, type):

    if(modal == 'text'):
        modelPath = textModelPath
    elif(modal == 'video'):
        modelPath = videoModelPath
    else:
        print("Invalied Input")
        return None

    ## Evaluation
    model = inter_model_load(modelPath)

    ### Convert labels to categorical one-hot encoding
    x = np.load('inputs/'+ modal + '_' + type + '.npy')

    ### Model build
    model.summary()

    feature = model.predict(x, verbose=1, batch_size=512)
    print(np.shape(feature))

    if not(os.path.isdir('features')):
        os.makedirs(os.path.join('features'))

    np.save('features/'+modal+ '_BN_' + type + '.npy', feature)

    print("Finished")

    K.clear_session()

    return None

if __name__ == '__main__':
    print("text/video:")
    modal = input()
    print("train/val/test1/test2/test3:")
    type = input()
    main(modal,type)

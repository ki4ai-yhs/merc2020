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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

textModelPath = 'inputs/text_model_acc_0.4523.h5'
videoModelPath = 'inputs/video_model_acc_0.2971.h5'

def inter_model_load(model_path):
    text_model=load_model(model_path)
    inter_layer_model = Model(inputs=text_model.input, outputs=text_model.get_layer('BottleNeck').output)
    return inter_layer_model

def main(modal):

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
    x_train = np.load('inputs/'+modal+'_train.npy')
    x_val = np.load('inputs/'+modal+'_val.npy')

    ### Model build
    model.summary()

    feature = model.predict(x_train,verbose=1,batch_size=512)
    print(np.shape(feature))
    np.save('features/'+modal+'_BN_train.npy',feature)

    feature = model.predict(x_val,verbose=1)
    print(np.shape(feature))
    np.save('features/'+modal+'_BN_val.npy',feature)

    print("Finished")

    return None

if __name__ == '__main__':
    print("text/video:")
    temp = input()
    main(temp)

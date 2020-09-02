import os
import numpy as np
import cv2
import csv
import time
import tensorflow as tf

# set gpu-memory usages
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from preprocessing import Preprocessor
from mtcnn import FaceCropperAll
import vggface

detectorType = 'mtcnn'
facecropper = FaceCropperAll(type=detectorType, resizeFactor=1.0)
preprocessor = Preprocessor()

def testInput(parentPath, interval=10):
    '''
    test for sample input(N x W x H x C)

    N: Number of input frames
    W: width of image
    H: height of image
    C: number of channel in image(3)

    :return: M x N x W x H x C inputs
    '''
    frameSeq = os.listdir(parentPath)
    frameSeq.sort()

    InputList = []
    cnt = 0
    
    for i in range(0, len(frameSeq), interval):

        frameSeqAbsPath = os.path.join(parentPath, frameSeq[i])
        try:
            img = cv2.imread(frameSeqAbsPath)
            InputList.append(img)
        except:
            break

    return InputList

def testFaceCropper(inputList):
    global facecropper

    res, cnt = facecropper.detectMulti(inputList)
    if(res is None):
        print('TEST FAIL FACE CROPPER')
        return None
    return res, cnt

def testPreprocessing(inputList):
    global preprocessor

    res = preprocessor.process(inputList)
    if(res is None):
        print('TEST FAIL PREPROCESSOR')
        return None
    return res

def main(frame_path):

    N = 10 # Use 1 frame for 1 second (10 frame/sec)
    img = testInput(frame_path, N)
    
    # face cropper
    resCropped, cnt = testFaceCropper(img)
    print("Face cropped:",cnt)

    # not detected
    if cnt == 0:
        return np.zeros((1,4096))

    # pre-processing
    resPreprop = testPreprocessing(resCropped)

    # VGGface feature
    video_feature=vggface.extractFeature([resPreprop])
 
    return video_feature

if __name__ == '__main__': 

    print("train/val:")
    select = input()
    
    if(select == "train"):
        seqPath = 'dataset/image_frame_train/'
        csvPath = 'dataset/qia_train.csv'
        savePath = 'features/train/'

    elif(select == "val"):
        seqPath = 'dataset/image_frame_val/'
        csvPath = 'dataset/qia_val.csv'
        savePath = 'features/val/'

    with open(csvPath) as csvfile:
        rdr = list(csv.DictReader(csvfile))
        for i in range(len(rdr)):
            file_ID = rdr[i]['FileID'] # file ID
            img_num = str(file_ID).zfill(5)
            img_path = seqPath + img_num
            save_path = savePath + img_num + '.npy'

            if(os.path.isfile(save_path)):
                print("already done:", i)
            else:
                print("data number:",i)
                features = main(frame_path=img_path)
                np.save(save_path, features)

## Reference: https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
import os

from facenet_pytorch import MTCNN
from PIL import Image

import numpy as np
import cv2
'''
Face Cropper total
'''

class FaceCropperAll():
    def __init__(self, marginRatio=1.3, type='mtcnn', resizeFactor=0.5):
        '''
        initializer of face cropper all
        :param marginRatio: ratio of margin
        :param type: type of face cropper:('mtcnn', 'haar' )
        '''

        if(type == 'mtcnn'):
            self.detector = MTCNNFaceDetector()
        else:
            assert False, 'Wrong face cropper type...'

    def detectMulti(self, inputList):
        return self.detector.detectMulti(inputList)

'''
Basic face cropper
'''
class FaceCropper():
    def __init__(self, marginRatio=1.3, resizeFactor=1.0):
        '''
        FaceCropper Basic Class
        :param marginRatio: margin of face(default: 1.3)
        '''
        self.marginRatio=1.3
        self.prevX = 0
        self.prevY = 0
        self.prevW = 0
        self.prevH = 0
        self.resizeFactor = resizeFactor

    def cropFace(self, input, x,y,w,h):
        '''
        Crop Face with given bbox
        :param input: input image
        :param x: X
        :param y: Y
        :param w: W
        :param h: H
        :return: cropped image
        '''

        x_n = int(x - (self.marginRatio - 1) / 2.0 * w)
        y_n = int(y - (self.marginRatio - 1) / 2.0 * h)
        w_n = int(w * self.marginRatio)
        h_n = int(h * self.marginRatio)

        return input[y_n:y_n + h_n, x_n:x_n + w_n],x,y,w,h

    def detect(self, input):
        '''
        Face detect with single input
        :param input: single image (W x H x C)
        :return: bbox information(x,y,w,h)
        '''
        pass
    def detectMulti(self, inputList):
        '''
        Face detect with multiple inputs
        :param inputList: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C)
        '''
        pass

'''
MTCNN Face Detector
'''
class MTCNNFaceDetector(FaceCropper):
    def __init__(self, marginRatio=1.3, resizeFactor=1.0):
        super().__init__(marginRatio, resizeFactor)
        # load mtcnn model
        # self.mtcnnModel = MTCNN()
        self.mtcnn = MTCNN(select_largest=True, device='cuda:1')

    def detect(self, img):
        '''
        Face detect with single input
        :param img: single image (W x H x C)
        :return: face croppped image (W' x H' x C)
        :cnt: number of successfully detected faces
        '''

        imgResize = cv2.resize(img, dsize=(0,0), fx=self.resizeFactor, fy=self.resizeFactor)

        res = self.mtcnnModel.detect_faces(imgResize)

        # if no faces detect, than return previous bbox position
        if(len(res) == 0):
            return self.prevX, self.prevY, self.prevW, self.prevH, False

        # process of margin ratio
        x,y,w,h = res[0]['box']

        # save the previous result
        self.prevX = x
        self.prevY = y
        self.prevW = w
        self.prevH = h

        return x,y,w,h, True

    def detectMulti(self, inputList):
        '''
        Face detect with multiple inputs
        :param inputList: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C)

        resList = []
        cnt=0
        
        for eachInput in inputList:
            bbox = self.detect(eachInput)
            if(bbox is None):
                return None
            x,y,w,h, isDetected = bbox

            if(isDetected == True):
                cnt+=1

            res = self.cropFace(eachInput, x,y,w,h)

            resList.append(res[0])
        return np.array(resList), cnt

        '''

        imgList = []
        resList = []
        for eachInput in inputList:
            imgResize = cv2.resize(eachInput, dsize=(0,0), fx=self.resizeFactor, fy=self.resizeFactor)
            img8U = np.uint8(imgResize)
            img = cv2.cvtColor(img8U, cv2.COLOR_BGR2RGB)
            imgList.append(Image.fromarray(img))

        boxes, _ = self.mtcnn.detect(imgList)
        cnt = 0

        for i in range(boxes.shape[0]):
            if (boxes[cnt] is None):
                continue
            else:
                box = boxes[cnt]
                x = box[0][0]
                y = box[0][1]
                w = box[0][2]
                h = box[0][3]
                res = self.cropFace(eachInput, x,y,w,h)
                resList.append(res[0])
                cnt+=1

        print("cnt:",cnt)

        return resList, cnt



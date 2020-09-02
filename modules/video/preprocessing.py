'''
Preprocessor for input sequences
'''

from keras.applications.inception_v3 import preprocess_input
import cv2
import numpy as np
class Preprocessor():
    def __init__(self):
        pass
    def process(self, inputList, resizedFactor=(224,224)):
        resList = []
        for eachInput in inputList:
            try:
                eachInputResized = cv2.resize(eachInput, resizedFactor)
                eachInputPreprocessed = preprocess_input(eachInputResized)
            except:
                eachInputPreprocessed = np.zeros((224,224,3))

            resList.append(eachInputPreprocessed)
        return resList



from keras.models import load_model
import numpy as np
import csv


if __name__ == '__main__':

    MODEL_NAME = 'model/multimodal_model_acc_0.5256.h5'

    print('test1/test2/test3')
    tmp = input()

    TEXT_DATA = 'features/text_BN_' + tmp + '.npy'
    SPEECH_DATA = 'features/video_BN_' + tmp + '.npy'
    VIDEO_DATA = 'features/speech_BN_' + tmp + '.npy'

    emotion_list = ['hap', 'ang', 'dis', 'fea', 'neu', 'sad', 'sur']

    model = load_model(MODEL_NAME)

    t = np.load(TEXT_DATA)
    v = np.load(VIDEO_DATA)
    s = np.load(SPEECH_DATA)

    output = np.argmax(model.predict([t, v, s], verbose = 1 , batch_size = 256), axis = 1)

    results = []
    for i in range(len(output)):
        results.append(emotion_list[output[i]])

    print(results)
    with open(tmp + '.csv', 'w') as f:
        for line in results:
            f.write(line)
            f.write('\n')


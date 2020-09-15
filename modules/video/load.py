import numpy as np
import glob

##Set root path##

def load_video_data(folder_path,type):

    _MAX_LEN = 6

    if(type=='train'):
        feature_path = sorted(glob.glob(folder_path + 'train/' + '*.npy'))
    elif(type=='val'):
        feature_path = sorted(glob.glob(folder_path + 'val/' + '*.npy'))
    elif(type=='test1'):
        feature_path = sorted(glob.glob(folder_path + 'test1/' + '*.npy'))
    elif(type=='test2'):
        feature_path = sorted(glob.glob(folder_path + 'test2/' + '*.npy'))
    elif(type=='test3'):
        feature_path = sorted(glob.glob(folder_path + 'test3/' + '*.npy'))
    else:
        print("Invalied Input")

    output = np.zeros((len(feature_path), _MAX_LEN, 4096))

    for i in range(len(feature_path)):
        
        F = np.load(feature_path[i])

        if (F.shape[0] <= _MAX_LEN):
            output[i,:F.shape[0],:F.shape[1]] = F
        else:
            output[i,:F.shape[0],:F.shape[1]] = F[:_MAX_LEN,:]

        print("data number:", i)

    print(np.shape(output))

    np.save(folder_path + 'video_' + type + '.npy', output)

    return output

image_folder_path = 'features/'

load_video_data(image_folder_path, 'test1')
load_video_data(image_folder_path, 'test2')
load_video_data(image_folder_path, 'test3')

import numpy as np
import glob

## Set root path
speech_folder_path = "dataset/speech_features/"

def speech_data(folder_path,type):

    if(type=='train'):
        feature_path = sorted(glob.glob(folder_path + 'train/' + '*.npy'))
    
    elif(type=='val'):
        feature_path = sorted(glob.glob(folder_path + 'val/' + '*.npy'))
    else:
        print("Invalied Input")

    if not feature_path:
        print("No data")
        return None

    output = np.zeros((len(feature_path),64))    
    for i in range(len(feature_path)):
        output[i]=np.load(feature_path[i])
        print(i)

    np.save('speech_'+type+'.npy',output)

    return output

speech_data(speech_folder_path,'train')
speech_data(speech_folder_path,'val')

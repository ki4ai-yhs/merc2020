import numpy as np
import glob

##Set root path
text_folder_path = "dataset/text_embed/"

def text_data(folder_path,type):

    _MAX_LEN = 30

    if(type=='train'):
        feature_path = sorted(glob.glob(folder_path + 'train/' + '*.npz'))

    elif(type=='val'):
        feature_path = sorted(glob.glob(folder_path + 'val/' + '*.npz'))
    
    else:
        print("Invalied Input")

    output = np.zeros((len(feature_path), _MAX_LEN, 200))

    for i in range(len(feature_path)):
        F = np.load(feature_path[i])['word_embed']
        if (F.shape[0] <= _MAX_LEN):
            output[i,:F.shape[0],:F.shape[1]] = F
        else:
            output[i,:F.shape[0],:F.shape[1]] = F[:_MAX_LEN,:]
        
        print("data number:",i)

    np.save('dataset/text_'+type+'.npy',output)

    return output

text_data(text_folder_path,'train')
text_data(text_folder_path,'val')

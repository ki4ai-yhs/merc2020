import csv
import numpy as np
 


emotion = ['hap', 'ang', 'dis', 'fea', 'neu', 'sad', 'sur']

def label_to_npy(load_dir, save_dir):

    label = []
    with open(load_dir) as csvfile:
        reader = csv.DictReader(csvfile)
        for rdr in reader:
            emo = rdr["Emotion"]
            print(emo)
            for i in range(len(emotion)):
                if(emotion[i] == str(rdr["Emotion"])):
                    label.append(i)

    label = np.array(label)
    np.save(save_dir, label)
    print("Done!")

    return None


if __name__ == "__main__":
    load_dir = '../dataset/qia2020/qia_train.csv'
    save_dir = '../dataset/qia2020/train_labels.npy'
    label_to_npy(load_dir, save_dir)

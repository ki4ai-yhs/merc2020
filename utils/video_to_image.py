import cv2
import csv
import os
import glob
import threading


def video_to_frame(load_dir,save_dir):
    video_file_path = sorted(glob.glob(load_dir))

    for i in range(len(video_file_path)):

        cap = cv2.VideoCapture(video_file_path[i])

        name = video_file_path[i].split('/')[-1].split('.')[0]

        save_folder_path = save_dir + '/'+ str(name[:5])

        print(i, name)

        if not (os.path.isdir(save_folder_path)):
            os.makedirs(os.path.join(save_folder_path))

        count = 0
        fps = 30
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if(count%fps==0):
                    save_file_path =  save_folder_path +"/" + name + "_{:d}.jpg".format(count)
                    cv2.imwrite(os.path.join(save_file_path), frame)
                count += 1
            else:
                break


if __name__ == "__main__":

    load_dir = '../dataset/qia2020/train/*.mp4'
    save_dir = "../dataset/qia2020/image_frame_train/"
    video_to_frame(load_dir, save_dir)
    

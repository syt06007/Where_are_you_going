import os
import cv2
import numpy as np

root = 'data/raw/20231105_145904.mp4'
vid_path1 = root

path_lst = [vid_path1]

for vid_path in path_lst:
    print(f'----VIDEO PATH : {vid_path}----')
    video = cv2.VideoCapture(vid_path)
    
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print('length :', length)
    print('WIDTH :', width)
    print('height :', height)
    print('fps :', fps)

    try:
        new_name = vid_path[:-4]
        os.makedirs(new_name)
    except OSError:
        print(print ('Error: Creating directory. ' +  new_name))
    print(f'----NEW NAME : {new_name}----')
    count = 0

    while video.isOpened():
        ret, img = video.read()
        if ret == False:
            break
        if int(video.get(1)) % 1 == 0 :
            cv2.imwrite('data/images' + f'/frame{str(count).zfill(5)}.jpg', img)
            print('Saved frame number :', str(int(video.get(1))))
            count += 1

    video.release()

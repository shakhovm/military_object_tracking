import cv2
import os
import sys


path = sys.argv[1]

videos = list(filter(lambda x: x.endswith(".mp4"), os.listdir(path)))
videos.sort(key=lambda f: os.stat(os.path.join(path, f)).st_size, reverse=False)
for video in videos:
    i = 1
    video_path = os.path.join(path, video)
    cap = cv2.VideoCapture(video_path)
    dir_path = video_path[:-4]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # print(dir_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        number_len = len(str(i))

        img_name = "00000000"[:-number_len] + str(i) + ".jpg"
        # print(img_name)
        cv2.imwrite(os.path.join(dir_path, img_name), frame)
        i += 1
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break

import cv2
import numpy as np
import os

def main():

    video_path = './demo_video/test_2.mp4'
    save_path  = video_path.replace('.mp4','') + '/'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite(save_path + str(count) + '.png', frame)
        count += 1
        if ret == True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
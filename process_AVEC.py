import os
import shutil
import cv2

def main():

    video_path        = './data/AVEC14/Video'
    label_path        = './data/AVEC14/Label'
    image_path        = checkPath('./data/AVEC14/Image', True)

    ignore_data = '.DS_Store'
    # mode        = ['Training', 'Development', 'Testing']
    

    for mode in os.listdir(video_path):
        if ignore_data not in mode:

            i_mode = checkPath(os.path.join(image_path, mode))
            v_mode = os.path.join(video_path, mode)
            
            for dataType in os.listdir(v_mode):
                if ignore_data not in dataType:
                    i_type = checkPath(os.path.join(i_mode, dataType))
                    v_type = os.path.join(v_mode, dataType)

                    for v_file in os.listdir(v_type):
                        if ignore_data not in v_file:
                            i_file_dir = v_file.replace('.mp4', '')
                            i_file_dir = checkPath(os.path.join(i_type, i_file_dir))
                            v_file     = os.path.join(v_type, v_file)

                            readVideo(v_file, i_file_dir)
                            print(v_file, '=>', i_file_dir, 'done')

def readVideo(path, save_dir):

    cap = cv2.VideoCapture(path)
    if (cap.isOpened() == False):
        print('error =>', path)
    
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image_path = os.path.join(save_dir, str(count) + '.png')
            cv2.imwrite(image_path, frame)
            count += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

def checkPath(path, is_del = False):

    if os.path.isdir(path):
        if is_del:
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)
    return path

if __name__ == "__main__":
    main()
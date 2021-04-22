import os

def main():

    path = './data/CAER/CAER/CAER/'
    for type_dir in os.listdir(path):
        # train, validation, test
        txt_file = path + type_dir + '.txt'
        type_dir = os.path.join(path, type_dir)
        for label_dir in os.listdir(type_dir):
            # Anger, Disgust, Fear, Happy, Neutral, Sad, Suprise
            label = label_dir
            label_dir = os.path.join(type_dir, label_dir)
            for avi_file in os.listdir(label_dir):
                avi_file = os.path.join(label_dir, avi_file)
                f = open(txt_file, 'a')
                f.write(avi_file + ' ' + label + '\n')
                f.close()
if __name__ == "__main__":
    main()
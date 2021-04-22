import os
import numpy as np
import shutil

def main():

    path = './data/AVEC14/VAD/Testing'
    current = 'AffectLabels'
    current = os.path.join(path, current)
    ignore_data = '.DS_Store'
    for filename in os.listdir(current):
        src = os.path.join(current, filename)

        target = filename.split('-')[-1].replace('.csv', '')

        target = os.path.join(path, target)

        if ignore_data not in src:
            target = os.path.join(target, filename)

            print(src, target)
            shutil.copyfile(src, target)

if __name__ == "__main__":
    main()
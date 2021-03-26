import os
from multiprocessing import Pool
import cv2
import argparse


def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def main(args):
    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        path_data = os.path.join(args.data_root, seq_name)
        path_vdo = os.path.join(path_data, 'vdo.avi')
        path_images = os.path.join(path_data, 'img1')
        check_and_create(path_images)

        vidcap = cv2.VideoCapture(path_vdo)
        success, image = vidcap.read()

        count = 1
        output_iterator = []
        name_iterator = []
        while success:
            path_image = os.path.join(path_images, '%06d.jpg' % count)
            success, image = vidcap.read()
            if success: 
                output_iterator.append(image)
                name_iterator.append(path_image)
            count += 1
            if count%512==0 or success==False: 
                with Pool(16) as p:
                    p.map(save, zip(output_iterator, name_iterator) )
                output_iterator = []
                name_iterator = []

def save(inputs ):
    image, name = inputs
    cv2.imwrite(name, image)
    print('Data path: %s' % name)


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Extract video frames')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)

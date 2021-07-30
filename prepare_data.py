import os
import cv2
import numpy as np
import tensorflow as tf

def prepare_data(folderpath):

    imgs_hr = []
    imgs_lr = []
    fails = 0

    for f in os.listdir(folderpath):
        img_hr = cv2.imread(os.path.join(folderpath,f))
        img_hr = np.asarray(img_hr, dtype=float)

        try:
          img_hr = tf.image.random_crop(img_hr, [96, 96, 3])
          img_lr = tf.image.resize(img_hr, [24, 24], method='bicubic', antialias=True)

          img_hr = (img_hr-127.5)/127.5
          img_lr = img_lr/255.0

          imgs_hr.append(img_hr)
          imgs_lr.append(img_lr)
        except:
          fails = fails +1

    imgs_hr = tf.convert_to_tensor(imgs_hr)
    imgs_lr = tf.convert_to_tensor(imgs_lr)

    return imgs_lr, imgs_hr
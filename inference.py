import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage import io #scikit-image
import math
import tqdm
import seaborn as sns

im_width = 384
im_height = 384

my_parser = argparse.ArgumentParser()

my_parser.add_argument('--img', action='store',required=True)

args = my_parser.parse_args()

image_path=args.img


smooth=100

from math import log10, sqrt
import cv2
import numpy as np

def signaltonoise(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0): # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection+100) / (np.sum(y_true) + np.sum(y_pred_bin)+100)

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def iou_val(y_true,y_pred):
    intersection=np.sum(y_true*y_pred)
    sum_=np.sum(y_true+y_pred)
    return (intersection + 100) / (sum_ - intersection + 100)

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def unet(input_size=(256,256,3)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])
img = cv2.imread(image_path)
model = unet()
model = load_model('unet_veins.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef,'signaltonoise':signaltonoise})
#img = cv2.imread(image_path)
img1=cv2.imread(image_path)
img = cv2.resize(img ,(im_height, im_width))
img1 = cv2.resize(img1 ,(im_height, im_width))
img = img / 255
img=np.expand_dims(img, axis=0)
#img = img[np.newaxis, :, :, :]
pred=model.predict(img)
Y_pred=np.round(pred)
Y_val=np.round(np.array(img))
FP = len(np.where(Y_pred - Y_val  == -1)[0])
FN = len(np.where(Y_pred - Y_val  == 1)[0])
TP = len(np.where(Y_pred + Y_val ==2)[0])
TN = len(np.where(Y_pred + Y_val == 0)[0])
cmat = [[TP, FN], [FP, TN]]
print(cmat)
sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("cnf_{}".format(image_path))
plt.show()

# plt.figure(figsize=(12,12))
# plt.subplot(1,2,1)
# plt.imshow(np.squeeze(img1))
# plt.title('Original Image')
# plt.subplot(1,2,2)
# plt.imshow(np.squeeze(pred) > .5)
# plt.title('Prediction')
# plt.show()
# op=np.squeeze(pred)*255
# op=cv2.resize(op,(im_height, im_width))
# cv2.imwrite('op_'+image_path,op)


# for files in sorted(os.listdir('images')):
# 	img=cv2.imread(os.path.join('images',files))
# 	img1=img
# 	img1 = cv2.resize(img1 ,(im_height, im_width))
# 	img = cv2.resize(img ,(im_height, im_width))
# 	img = img / 255
# 	img=np.expand_dims(img, axis=0)
# 	pred=model.predict(img)
# 	# cv2.imwrite(f'all/{files}',np.squeeze(pred)*255)
# 	plt.figure(figsize=(12,12))
# 	plt.subplot(1,3,1)
# 	plt.imshow(np.squeeze(img))
# 	plt.title('Original Image')
# 	plt.subplot(1,3,2)
# 	plt.imshow(np.squeeze(cv2.imread(os.path.join('masks',files))))
# 	plt.title('Original Mask')
# 	plt.subplot(1,3,3)
# 	plt.imshow(np.squeeze(pred) > .5)
# 	plt.title('Prediction')
# 	print(files)
# 	plt.show()

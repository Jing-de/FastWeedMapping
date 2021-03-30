import sys
import time
import shutil
import numpy as np
import glob

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from PIL import Image
import scipy.sparse

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import models
#from models import setup_model_resnet_manual_highres_center_only
#from models import setup_model_resnet_manual_highres
#from models import setup_model_resnet_manual_veryhighres_center_only_small
#from models import setup_model_resnet_manual_veryhighres_center_only_very_small
#from models import setup_model_resnet_manual_veryhighres_center_only
#from models import setup_model_resnet_manual_highres
#from models import setup_model_resnet_manual_highres_low
#from models import setup_model_resnet_manual_highres_lower
#from models import setup_model_resnet_manual_highres_lowest
#from models import setup_model_resnet_manual_highres
#from models import setup_model_resnet_manual_highres_center_only

from matplotlib import pyplot as plt
from IPython.display import clear_output

import keras
from keras.callbacks import TensorBoard, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

use_coordinates = False
# GLOBAL CONSTANTS
TRAINSET_DIR = './ImageTrainSet/201x201_TrainSetLeftRight'
#VALIDATION_DIR = './ImageTrainSet/201x201_ValSetLeftRight_10' # IT WONT BE USED
OUTPUT_DIR = './Tmp'
TRAINING_METRICS_OUTPUT_DIR = './Results/Training_Metrics'
SAVED_MODEL_OUTPUT_DIR = './Results/Saved_Model_As_h5'

# PARAMETERS
LR = 0.01
DECAY = 0
TRAIN_IMAGE_SIZE = (201, 201, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 100

def generate_setup_dict():
    from models import setup_model_resnet_manual_highres_center_only


    setup_per_filename = {
    'resnet_manual_highres_center_only_f1_2_f2_4':(setup_model_resnet_manual_highres_center_only, 2, 4),
    'resnet_manual_highres_center_only_f1_4_f2_8':(setup_model_resnet_manual_highres_center_only, 4, 8),
    'resnet_manual_highres_center_only_f1_6_f2_12':(setup_model_resnet_manual_highres_center_only, 6, 12),
    'resnet_manual_highres_center_only_f1_8_f2_16':(setup_model_resnet_manual_highres_center_only, 8, 16),
    'resnet_manual_highres_center_only_f1_10_f2_20':(setup_model_resnet_manual_highres_center_only, 10, 20),
    'resnet_manual_highres_center_only_f1_12_f2_24':(setup_model_resnet_manual_highres_center_only, 12, 24),
    'resnet_manual_highres_center_only_f1_14_f2_28':(setup_model_resnet_manual_highres_center_only, 14, 28),
    'resnet_manual_highres_center_only_f1_16_f2_32':(setup_model_resnet_manual_highres_center_only, 16, 32),
    'resnet_manual_highres_center_only_f1_32_f2_64':(setup_model_resnet_manual_highres_center_only, 32, 64),

    }
    return setup_per_filename

def get_classes():
    """  Gives the name of each class
    Arguments:
        No args
    Returns:
        label_names (list)
    """
    label_names = ["MAT","MON","PAP","SOIL","VER","VIO"] # this one is to be used with "TrainSetLeftRight"
    #label_names = ["MATCH_","PAPRH_","SOIL__","TRZAW_","VERHE_","VIOAR_"] # this one is to be used with "TrainSetLeftRight_90"
    return label_names

def coordinate_layer(image):
    mesh_x,mesh_y = np.meshgrid(np.arange(-100,101), np.arange(-100,101))
    coords = np.sqrt(np.square(mesh_x) + np.square(mesh_y))/1000.0
    return coords
    

def load_images_folder(path, max_images=99999999999):
    pattern = path+"/*.tif"
    files = sorted(glob.glob(pattern))[0:max_images]
    print("Loading images from path %s..." % (pattern))

    files_loaded = []
    if use_coordinates:
        images = np.zeros((len(files), 201, 201, 4))
    else:          
        images = np.zeros((len(files), TRAIN_IMAGE_SIZE[0], TRAIN_IMAGE_SIZE[1], 3))
    discarded = 0
    loaded = 0
    for i in range(0, len(files)):
        if i % 1000 == 0:
            print(i)
        image = np.array(Image.open(files[i]))/255.0
        if image.shape[0:2] == images[0, :, :, :].shape[0:2]:
            if use_coordinates:
                images[loaded, :, :, 0:3] = image
                images[loaded, :, :, 3] = coordinate_layer(image)
            else:
                images[loaded, :, :, :] = image
            loaded += 1
            files_loaded.append(files[i])
        else:
            #print("Warning: discarding image %s which is of size %s" % (files[i],image.shape))
            discarded += 1

    images = images[0:loaded, :, :, :]
    images = np.float16(images)
    
    print("Loaded %i images from path %s [discarded %i images which are not of correct size]" % (loaded, path, discarded))
    return images, files_loaded


def predict_image_set_with_augmentation(images, model):
    augmented_images,_ = augment(images, np.zeros(images.shape[0]))
    predictions = model.predict(augmented_images)
    return np.mean(predictions, axis=0, keepdims=True)

    
def augment(images, labels=None):
    if images.shape[0] > 1:
        show_progress = True
    else:
        show_progress = False
    if show_progress:
        print("Augmenting %i images..." % images.shape[0])
    images_augmented = np.zeros((images.shape[0]*8, images.shape[1], images.shape[2], images.shape[3]),"float16")
    if not labels is None:
        labels_augmented = np.zeros(images.shape[0]*8)
    for i in range(0,images.shape[0]):
        if i % 1000 == 0 and show_progress:
            print(i)
        image = images[i,:,:,:]
        for j in range(0,8):
            k = j % 4
            new_image = np.rot90(image,k)
            if int(j/4) % 2 == 1:
                new_image = np.flipud(new_image)
            images_augmented[i*8+j,:,:,:] = new_image
            if not labels is None:
                labels_augmented[i*8+j] = labels[i]
                
    if show_progress:
        print("Done.")
    if not labels is None:
        return images_augmented, labels_augmented
    else:
        return images_augmented
    

def shuffle(images,labels,files):
    np.random.seed(1)
    perm = np.random.permutation(images.shape[0])
    images = images[perm,: , :, :]
    labels = labels[perm]
    files = [files[i] for i in perm]
    return images, labels, files
    

def load_images_classes(path, max_num_images=99999999999):
    classes = get_classes()
    images = None
    files = []
    for i in range(0, len(classes)):
        images_class, files_class = load_images_folder(path + "/" + classes[i], max_num_images)
        labels_class = i*np.ones(images_class.shape[0])     
        files.extend(files_class)
        if images is None:
            images = images_class
            labels = labels_class
        else:
            images = np.concatenate((images, images_class))
            labels = np.concatenate((labels, labels_class))
    images,labels,files = shuffle(images, labels, files)
    return images, labels, files


from pyimagesearch.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from PIL import Image
import glob
import random

# load the pre-trained CNN from disk
print("[INFO] loading model...")
model = tf.keras.models.load_model("./Models/Keras_h5/resnet_manual_highres_center_only_f1_10_f2_20.h5")

image_files = glob.glob("./201x201_TestSetMid_selected/*/*.tif")
random.seed(1)
#image_files = random.sample(image_files, 1500)
k=0
print('{} files will be analyzed.'.format(len(image_files)))
for image_file in image_files:
    try:
        # load the original image from disk (in OpenCV format) and then
        # resize the image to its target dimensions
        orig = cv2.imread(image_file)
        resized = cv2.resize(orig, (201, 201), interpolation=cv2.INTER_CUBIC)
        #resized = cv2.resize(orig, (224, 224))

        # load the input image from disk (in Keras/TensorFlow format) and
        # preprocess it
        image = np.array(Image.open(image_file))/255.0
        image = np.expand_dims(image, axis=0)


        # use the network to make predictions on the input imag and find
        # the class label index with the largest corresponding probability
        preds = model.predict(image)
        i = np.argmax(preds[0])

        # initialize our gradient class activation map and build the heatmap
        cam = GradCAM(model, i, layerName="res5b_branch2b")
        heatmap = cam.compute_heatmap(image)

        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

        # display the original image and resulting heatmap and output image
        # to our screen
        output = np.vstack([orig, heatmap, output])
        output = imutils.resize(output, height=700)
        cv2.imwrite(image_file.replace("201x201_TestSetMid_selected","201x201_TestSetMid_selected_annotated"),output)
        if k%50==0:
            print('the file has been saved at', image_file.replace("201x201_TestSetMid_selected","201x201_TestSetMid_selected_annotated"))
        k+=1
    except:
        pass
print('We generated {} annotated images.'.format(k))

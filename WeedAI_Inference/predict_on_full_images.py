import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import tensorflow as tf
import numpy as np
import sys
import keras
from keras.layers import Conv2D, AveragePooling2D
from keras.engine.topology import Layer
import keras.backend as K
import glob    
from PIL import Image, ImageDraw
import train_model
import utils
import csv
import evaluate_model
import time
import models
import pandas as pd


save_annotations = True
embedded_version = False
image_format = "JPG"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict images, generate confusion matrices and the annotation'
                                 'for the image segmentation.')
    parser.add_argument('-p', '--precision', type=str, required=True, 
                        help='desired precision mode of your model, either fp32 or fp16')
    parser.add_argument('-t', '--test', action='store_true', help='set True if you want to include test prefix')
    args = vars(parser.parse_args())
    return args
    
        
class FastPooling2D(Layer):
    """ Creates the Fast Pooling class to build what is called as `flex model`.
    The parameter `last_dim` determines the multiplicity on the number of filters.
    """
    def __init__(self, pool_size=1, last_dim=16, **kwargs):
        self.pool_size = pool_size
        self.last_dim = last_dim
        super(FastPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FastPooling2D, self).build(input_shape)

    def call(self, x):
        inputs = x
        inputs.set_shape((1, 4912/2/2, 3264/2/2, self.last_dim))
        
        # first dimension
        cum_sum_1 = tf.cumsum(inputs, axis=1)
        paddings = np.zeros((4,2), dtype=int)
        paddings[1,0] = self.pool_size
        paddings = K.variable(paddings, dtype='int32')
        x_padded = tf.pad(x, paddings)
        x_padded = x_padded[:, 0:-self.pool_size, :, :]
        cum_sum_2 = tf.cumsum(x_padded, axis=1)
        summed = cum_sum_1 - cum_sum_2
        inputs = summed[:,self.pool_size-1:,:,:]
        
        # second dimension
        cum_sum_1 = tf.cumsum(inputs, axis=2)
        paddings = np.zeros((4, 2), dtype=int)
        paddings[2,0] = self.pool_size
        paddings = K.variable(paddings, dtype='int32')
        x_padded = tf.pad(inputs, paddings)
        x_padded = x_padded[:, :, 0:-self.pool_size, :]
        cum_sum_2 = tf.cumsum(x_padded, axis=2)
        summed = cum_sum_1-cum_sum_2
        summed = summed[:, :, self.pool_size-1:, :]
        outputs = summed/(self.pool_size*self.pool_size)
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, None, None, input_shape[3])  
        

def annotate_and_save(annotations, d, image_file, model_name, precision_mode):
    base_image_file = image_file.replace('Results/seeds/annotations_trt', 'Data')
    img = Image.open(base_image_file)
    draw = ImageDraw.Draw(img, mode="RGBA")
    for i in range(0,len(annotations)):
        color,_ = utils.get_color_probs(annotations[i][0])
        if not color is None:
            x,y = (annotations[i][1], annotations[i][2])
            #draw.point((x,y),fill=color)
            draw.rectangle((x-d/2, y-d/2, x+d/2, y+d/2), fill=color)
    image_file_sv = image_file.replace('.' + image_format, '_annotated_by_' + precision_mode + '_' + model_name + '.jpg')
    img.save(image_file_sv)

    
def load_labels(image_file):
    base_image_file = image_file.replace('Results/seeds/annotations_trt', 'Data')
    label_file = base_image_file.replace('.' + image_format, '_SingleHerbsPos.txt') 
    labels = []
    with open(label_file, 'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        for row in tsvin:
            if not 'PosX' in row:
                x,y = int(row[0]),int(row[1])
                c = row[3]
                labels.append((c,x,y))
    return labels
  
    
def annotate_and_save_per_class(annotations, d, image_file, model_name, precision_mode):
    base_image_file = image_file.replace('Results/seeds/annotations_trt', 'Data')
    img = Image.open(base_image_file)
    classes = train_model.get_classes()
    labels = load_labels(image_file)
    classes_image = []
    
    for c in range(0,len(classes)):
        new_img = img.copy()
        draw = ImageDraw.Draw(new_img, mode='RGBA')
        for i in range(0,len(annotations)):
            color,predicted_class = utils.get_color_probs(annotations[i][0])
            classes_image.append(predicted_class)
            if predicted_class==c:
                x,y = (annotations[i][1], annotations[i][2])
                draw.rectangle((x-d/2, y-d/2, x+d/2, y+d/2), fill=color)
        for (c_name,x,y) in labels:
            solid_color = utils.get_color(classes.index(c_name),255)
            draw.ellipse((x-6, y-6, x+6, y+6), fill=solid_color)
            
        new_image_file = image_file.replace('.' + image_format, '_annotated_by_' + precision_mode + '_' + model_name + '_for_class_%s.jpg' % classes[c])
        new_img.save(new_image_file)
        print("Saved %s" % new_image_file)
        
    return classes_image


def predict_image(model, image_file, model_name='', precision_mode='FP32'):

    img = Image.open(image_file)
    sx,sy = img.size
    print("Predicting for image %s (%i x %i pixel)" % (image_file, sx, sy))
    d = 4
    annotations = []
    for x in range(0, sx-201, d):
        crops = []
        print(x)
        for y in range(0, sy-201, d):
            crop = img.crop((x, y, x+201, y+201))
            crops.append((crop, x+100, y+100))
        crops_np = np.zeros((len(crops), 201, 201, 3))
        for i in range(0,len(crops)):
            crops_np[i, :, :, :] = np.array(crops[i][0])/255.0
    
        print("Predicting on %i crops" % crops_np.shape[0])
        start = time.time()
        predictions = model.predict(crops_np)
        elapsed = time.time() - start
        print("Took %f seconds per crop" % (elapsed/float(crops_np.shape[0])))
        for i in range(0,len(crops)):
            annotations.append((predictions[i, :],crops[i][1], crops[i][2]))
    
    annotate_and_save(annotations, d, image_file, model_name, precision_mode)
    annotate_and_save_per_class(annotations, d, image_file, model_name, precision_mode)
    labels = load_labels(image_file)
    confusion_matrix = np.zeros((num_classes,num_classes))
    for (c_name,x,y) in labels:
        if 100 <= x < sx-101 and 100 <= y < sy-201:
            crop = img.crop((x-100, y-100, x+101, y+101))
            crop_np = np.array(crop)/255.0
            probs = model.predict(crop_np.reshape(1, 201, 201, 3))
            predicted_class = np.argmax(probs)
            c = train_model.get_classes().index(c_name)
            confusion_matrix[c, predicted_class] += 1

    return confusion_matrix


def predict_image_highres(frozen_graph, x_tensor, y_tensor, image_file, model_name='', precision_mode='FP32'):

    img = Image.open(image_file)
    sx,sy = img.size
    print("Image size is %i x %i" % (sx, sy)) # sx = 4912, sy = 3264
    print("Loading image %s" % image_file)
    img_np = np.array(img)/255.0
    print(img_np.shape)
    if embedded_version:
        img = None
    

    print("Predicting for image %s (%i x %i pixel)" % (image_file, sx, sy))
    

    if not embedded_version:
        with tf.Session(graph=frozen_graph) as sess:
            start = time.time()
            predictions_flex = sess.run(y_tensor, feed_dict={x_tensor:np.expand_dims(img_np, 0)})
            print('predictions_flex.shape')
            print(predictions_flex.shape)
            elapsed = time.time() - start
        del img_np
        print("Prediction took %f seconds (inference on full image)" % elapsed)

    print("Merging predictions")
    # merge the predictions on the quarter images
    predictions_flex_combined = np.zeros(predictions_flex.shape)

    elapsed = time.time() - start
    if embedded_version:
        print("Prediction took %f seconds (inference on split up image)" % elapsed)

    if embedded_version:
        predictions_flex = predictions_flex_combined

    if save_annotations:
        print("Computing annotations...")
        annotations = []
        d = 4
        for x in range(100, sx-101, d):
            for y in range(100, sy-101, d):
                x0 = int(round(float(x-100)/4) + 15)
                y0 = int(round(float(y-100)/4) + 15)
                probs_flex = np.squeeze(predictions_flex[0, y0, x0, :])
                annotations.append((probs_flex, x, y))
        annotate_and_save(annotations, d, image_file, model_name, precision_mode)
        classes_image = annotate_and_save_per_class(annotations, d, image_file, model_name, precision_mode)


    labels = load_labels(image_file)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for (c_name, x, y) in labels:
        if 100 <= x < sx-101 and 100 <= y < sy-101:
            x0 = int(round(float(x-100)/4)+15)
            y0 = int(round(float(y-100)/4)+15)
            probs_flex = np.squeeze(predictions_flex[0, y0, x0, :])

            predicted_class = np.argmax(probs_flex)
            c = train_model.get_classes().index(c_name)
            confusion_matrix[c, predicted_class] += 1
    if save_annotations:
        return confusion_matrix, classes_image
    else:
        return confusion_matrix

    
FROZEN_GRAPH_FILEPATH = '.Models/Frozen_graphs/flex_resnet_manual_highres_center_only_f1_2_f2_4_frozen_graph.pb'
def trt_frozen_graph_and_tensors(model_name, 
                                 frozen_graph_filepath=FROZEN_GRAPH_FILEPATH, 
                                 precision_mode='FP16'):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    """ Loads a Tensorflow frozen graph and changes its precision mode.
    You can either use FP32 or FP16. FP32 is the original precision mode,
    but will use TensorRT optimization.
    
    Args:
        model_name (str): The name of your model, eg, resnet_manual_highres_center_only_f1_2_f2_4
        frozen_graph_filepath (str): Path to where the frozen graph was saved
        precision_mode (str): either 'FP32' or 'FP16'
    Returns:
        (tuple): tuple containing:
            frozen_graph (tf.compat.v1.Session): Session containing the TRT graph
            x (tf.Tensor): Tensor containing the x data
            y (tf.Tensor): Tensor containing the y data
    """
    
    if precision_mode in ['FP32', 'FP16']:
        print('OPENING FROZEN GRAPH FOR MODEL {}'.format(model_name))
        with open(frozen_graph_filepath, 'rb') as f:
            frozen_graph_gd = tf.compat.v1.GraphDef()
            frozen_graph_gd.ParseFromString(f.read())

            if precision_mode == 'FP16':
                converter = trt.TrtGraphConverter(input_graph_def=frozen_graph_gd, 
                                                  nodes_blacklist=['local_dense/truediv'],
                                                  precision_mode=precision_mode, 
                                                  use_calibration=True, 
                                                  is_dynamic_op=True)
                del frozen_graph_gd
                print('Converting to {}'.format(precision_mode))
                frozen_graph = converter.convert()
                print('Conversion finished')
                
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.compat.v1.Session(graph=tf.Graph(), config=config) as sess:
            if precision_mode == 'FP32':
                frozen_graph = frozen_graph_gd
            tf.import_graph_def(frozen_graph)

            input_node = 'import/input1_1'
            output_node = 'import/local_dense/truediv'

            frozen_graph = sess.graph
            x = frozen_graph.get_tensor_by_name(input_node + ':0')
            y = frozen_graph.get_tensor_by_name(output_node + ':0')
            return frozen_graph, x, y
    else:
        return 'Error on the precision mode'

#TODO: AUTOMATIC ALLOCATE FOLDERS DEPENDING ON ITS PRECISION MODE
def main(seed, filter_, num_classes, setup, model_name, images_dir, precision_mode, test):
    """ The main method is responsible for running the generation of confusion matrices,
    the time taken for inferences, and save them. It iterates over all seeds and filters,
    in our case, seeds ranges from 1 to 5, and filters from 2-4 to 16-32.
    
    Args:
        seed (str): The seed that generated the model loaded
        filter_ (list): List of strings, containing  the first and second filter used, such as ['2','4']
        num_classes (int): Number of classes of our problem.
        setup (dict): Dictionary that maps the model setup to its filter size
        model_name (str): Name of model to be used, eg `resnet_manual_highres_center_only_f1_2_f2_4`
        images_dir (str): Path to the directory where the images are located
        precision_mode (str): precision_mode (str): either 'FP32' or 'FP16'
    Returns:
        None
    """
    f1, f2 = filter_
    model_name = 'flex_random_seed_{}_resnet_manual_highres_center_only_f1_{}_f2_{}'.format(seed, f1, f2)
    frozen_graph_filepath = './Models/Frozen_graphs/{}_{}/'.format(f1,f2) + model_name + '_frozen_graph.pb'
    frozen_graph, x_tensor, y_tensor = trt_frozen_graph_and_tensors(
        model_name=model_name, 
        frozen_graph_filepath=frozen_graph_filepath, 
        precision_mode=precision_mode
        )

    elapsed_time_full_dataset = []
    sum_of_confusion_matrices = np.zeros((6, 6))
    
    with tf.compat.v1.Session(graph=frozen_graph) as sess:
        for image_file in [img for img in os.listdir(images_dir) if img.endswith('.JPG')]:

            img = Image.open(images_dir + image_file)
            sx,sy = img.size

            print("Image size is %i x %i" % (sx,sy)) # sx = 4912, sy = 3264
            print("Loading image %s" % image_file)

            img_np = np.array(img)/255.0
            del img

            print("Predicting for image %s (%i x %i pixel)" % (image_file,sx,sy))

            start = time.time()
            predictions_flex = sess.run(y_tensor, feed_dict={x_tensor:np.expand_dims(img_np, 0)})
            elapsed = time.time() - start
            elapsed_time_full_dataset.append(elapsed)
            del img_np #deleting afterwards to not take the deleting time into account

            print("Prediction took %f seconds (inference on full image)" % elapsed)
            print("Merging predictions")
            # merge the predictions on the quarter images
            predictions_flex_combined = np.zeros(predictions_flex.shape)

            elapsed = time.time()-start
            if embedded_version:
                print("Prediction took %f seconds (inference on split up image)" % elapsed)

            if embedded_version:
                predictions_flex = predictions_flex_combined

            if save_annotations:
                print("Computing annotations...")
                annotations = []
                d = 4
                for x in range(100, sx-101, d):
                    for y in range(100, sy-101, d):
                        x0 = int(round(float(x-100)/4) + 15)
                        y0 = int(round(float(y-100)/4) + 15)
                        probs_flex = np.squeeze(predictions_flex[0, y0, x0, :])
                        annotations.append((probs_flex, x, y))

                if test: # add a prefix for test to not replace real experiments
                    model_name = 'TEST_' + model_name

                # saving annotations
                annotation_dir = images_dir.replace('Data', 'Results/seeds/annotations_trt') + image_file
                annotate_and_save(annotations, d, annotation_dir, model_name, precision_mode)
                classes_image = annotate_and_save_per_class(
                    annotations, 
                    d, 
                    annotation_dir, 
                    model_name, 
                    precision_mode
                )

            labels = load_labels(annotation_dir)
            confusion_matrix = np.zeros((num_classes, num_classes))
            for (c_name, x, y) in labels:
                if 100 <= x < sx-101 and 100 <= y < sy-101:
                    x0 = int(round(float(x-100)/4) + 15 )
                    y0 = int(round(float(y-100)/4) + 15)
                    probs_flex = np.squeeze(predictions_flex[0, y0, x0, :])

                    predicted_class = np.argmax(probs_flex)
                    c = train_model.get_classes().index(c_name)
                    confusion_matrix[c, predicted_class] += 1
            print(confusion_matrix)
            sum_of_confusion_matrices += confusion_matrix

    print(sum_of_confusion_matrices)
    sum_of_cm_fp = './Results/seeds/preds_trt/{}/{}_{}/sum_of_cm_'\
                    .format(precision_mode.lower(), f1,f2) + model_name + '_fp32.npy'
    elapsed_time_fp = './Results/seeds/elapsed_trt/{}/{}_{}/time_taken_'\
                    .format(precision_mode.lower(), f1,f2) + model_name + '_fp32.npy'


    np.save(sum_of_cm_fp, sum_of_confusion_matrices)
    np.save(elapsed_time_fp, elapsed_time_full_dataset)
    tf.reset_default_graph()   

    
if __name__ == "__main__":
    np.random.seed(1)
    NUM_CLASSES= 6
    SETUP = utils.generate_setup_dict()
    IMAGES_DIR = './Data/201x201_TestSetMidFull_all/'
    
    args = parse_arguments()
    PRECISION_MODE = args['precision'].upper()
    TEST = args['test'] if args['test'] else False
    
    for seed in ['1','2','3','4','5']:
        for filter_ in [['2', '4'], ['4','8'],['6','12'],['8','16'],['10','20'],['12','24'],['14','28'],['16','32']]:
            model_name = 'resnet_manual_highres_center_only_f1_{}_f2_{}'.format(filter_[0], filter_[1])
            main(seed=seed,
                 filter_=filter_,
                 num_classes=NUM_CLASSES, 
                 setup=SETUP, 
                 model_name=model_name, 
                 images_dir=IMAGES_DIR, 
                 precision_mode=PRECISION_MODE, 
                 test=TEST)
import os
import tensorflow as tf
import keras
import numpy as np
import train_model_mp
import utils
import csv
import time
from keras.layers import Conv2D
from keras.engine.topology import Layer
from PIL import Image, ImageDraw


# CONFIGS
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)


# OPTIONS
save_annotations = False
embedded_version = False
image_format = "JPG"


class FastPooling2D(Layer):
    """ Create custom layer called FastPooling2D to
    speed-up the model
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
        paddings = np.zeros((4,2),dtype=int)
        paddings[1,0] = self.pool_size
        paddings = K.variable(paddings,dtype="int32")
        x_padded = tf.pad(x,paddings)
        x_padded = x_padded[:,0:-self.pool_size,:,:]
        cum_sum_2 = tf.cumsum(x_padded,axis=1)
        summed = cum_sum_1-cum_sum_2
        inputs = summed[:,self.pool_size-1:,:,:]
        
        # second dimension
        cum_sum_1 = tf.cumsum(inputs, axis=2)
        paddings = np.zeros((4, 2),dtype=int)
        paddings[2,0] = self.pool_size
        paddings = K.variable(paddings,dtype="int32")
        x_padded = tf.pad(inputs, paddings)
        x_padded = x_padded[:, :, 0:-self.pool_size, :]
        cum_sum_2 = tf.cumsum(x_padded, axis=2)
        summed = cum_sum_1-cum_sum_2
        summed = summed[:, :, self.pool_size-1:, :]
        outputs = summed/(self.pool_size*self.pool_size)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, None, None, input_shape[3])
        
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 'x', printEnd = "\r"):
    """
    fill element: '█'
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_color_probs(probs):
    c = np.argmax(probs)
    alpha = 100
    return get_color(c, alpha), c
    
def get_color(c,alpha):
    if c == 0:
        color = (255, 0, 0, alpha)
    if c == 1:
        color = (0, 255, 0, alpha)
    if c == 2:
        color = (0, 0, 255, alpha)
    if c == 3:
        color = (255, 0, 255, alpha)
    if c == 4:
        color = (255, 255, 0, alpha)
    if c == 5:
        color = (0, 255, 255, alpha)
    if c == 6:
        color = (255, 180, 100, alpha)
    return color
    

def annotate_and_save(annotations, d, image_file):
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img, mode="RGBA")
    for i in range(0,len(annotations)):
        color,_ = get_color_probs(annotations[i][0])
        if not color is None:
            x,y = (annotations[i][1],annotations[i][2])
            draw.rectangle((x-d/2, y-d/2, x+d/2, y+d/2),fill=color)
    image_file_sv = image_file.replace("." + image_format, "_annotated_by_" + model_name + ".jpg")
    image_file_sv = image_file_sv.replace(img_path, result_path)
    img.save(image_file_sv)

def load_labels(image_file):
    label_file = image_file.replace("." + image_format, "_SingleHerbsPos.txt")
    labels = []
    with open(label_file,"r") as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        for row in tsvin:
            if not "PosX" in row:
                x,y = int(row[0]),int(row[1])
                c = row[3]
                labels.append((c,x,y))
    return labels
   
def annotate_and_save_per_class(annotations,d,image_file):
    img = Image.open(image_file)
    classes = train_model_mp.get_classes()
    labels = load_labels(image_file)
    classes_image = []
    
    for c in range(0,len(classes)):
        new_img = img.copy()
        draw = ImageDraw.Draw(new_img, mode="RGBA")
        for i in range(0,len(annotations)):
            color,predicted_class = get_color_probs(annotations[i][0])
            classes_image.append(predicted_class)
            if predicted_class==c:
                x,y = (annotations[i][1],annotations[i][2])
                draw.rectangle((x-d/2,y-d/2,x+d/2,y+d/2),fill=color)
        for (c_name,x,y) in labels:
            solid_color = get_color(classes.index(c_name),255)
            draw.ellipse((x-6, y-6, x+6, y+6), fill=solid_color)
            
        new_image_file = image_file.replace("." + image_format,
                                            "_annotated_by_" + model_name + "_for_class_%s.jpg" % classes[c])
        new_image_file = new_image_file.replace(img_path,result_path)
        new_img.save(new_image_file)
        print("Saved %s" % new_image_file)
        
    return classes_image

def predict_image(model, image_file):

    img = Image.open(image_file)
    sx,sy = img.size
    print("Predicting for image %s (%i x %i pixel)" % (image_file,sx,sy))
    d = 4
    annotations = []
    for x in range(0,sx-201,d):
        crops = []
        print(x)
        for y in range(0,sy-201,d):
            crop = img.crop((x, y, x+201, y+201))
            crops.append((crop, x+100, y+100))
        crops_np = np.zeros((len(crops), 201, 201, 3))
        for i in range(0,len(crops)):
            crops_np[i,:,:,:] = np.array(crops[i][0])/255.0
    
        print("Predicting on %i crops" % crops_np.shape[0])
        start = time.time()
        predictions = model.predict(crops_np)
        elapsed = time.time()-start
        print("Took %f seconds per crop" % (elapsed/float(crops_np.shape[0])))
        for i in range(0,len(crops)):
            annotations.append((predictions[i,:],crops[i][1],crops[i][2]))
    
    annotate_and_save(annotations, d, image_file)
    annotate_and_save_per_class(annotations, d, image_file)
    labels = load_labels(image_file)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for (c_name,x,y) in labels:
        if 100 <= x < sx-101 and 100 <= y < sy-201:
            crop = img.crop((x-100, y-100, x+101, y+101))
            crop_np = np.array(crop)/255.0
            probs = model.predict(crop_np.reshape(1, 201, 201, 3))
            predicted_class = np.argmax(probs)
            c = train_model_mp.get_classes().index(c_name)
            confusion_matrix[c, predicted_class] += 1

    return confusion_matrix


def transform_highres_center_model(model, num_classes, model_name=None, from_model_to_setup=None):
    """Given a dictionary mapping the model name to a setup,
        creates the transformed model
    
    @Args:
        model (Keras model) : already loaded Keras model containing the original model to be transformed
        model_name (String) : the ID of the model (such as Wdai001.h5) contained in the model_times.csv
        from_model_to_setup (dict) : a mapping from model name that relates to its setup
        num_of_classes (int) : the number of target labels of the final model
    
    @Returns:
        flex_model (Keras model) : the transformed model
    """
    print("Transforming model {}...".format(model_name))
    filtersize1 = from_model_to_setup[model_name][1]
    filtersize2 = from_model_to_setup[model_name][2]
    flex_model, _ = from_model_to_setup[model_name][0]((None,None,3), num_classes, filtersize1, filtersize2)

    for i in range(1, len(flex_model.layers)):
        try:
            flex_model.layers[i].set_weights(model.layers[i].get_weights())
        except:
            pass

    before_pooling = flex_model.get_layer("before_pooling").output
    local_pooling = FastPooling2D(pool_size=20, name="local_pooling", last_dim=filtersize2)(before_pooling)
    local_dense = Conv2D(filters=6, kernel_size=1, activation="softmax", name="local_dense")(local_pooling)
    flex_model = keras.Model(inputs=flex_model.inputs, outputs=[local_dense])

    weights0 = model.get_layer("fc2").get_weights()
    weights0[0] = np.reshape(weights0[0], (1, 1, weights0[0].shape[0], weights0[0].shape[1]))
    flex_model.layers[-1].set_weights(weights0)
    print('transformation of base model into flex model done.')
    return flex_model


def predict_image_highres(model, flex_model, image_file):
    img = Image.open(image_file)
    sx,sy = img.size
    print("Image size is %i x %i" % (sx,sy)) # sx = 4912, sy = 3264
    print("Loading image %s" % image_file)
    img_np = np.array(img)/255.0
    print(img_np.shape)

    if embedded_version:
        img = None

    print("Predicting for image %s (%i x %i pixel)" % (image_file,sx,sy))

    if not embedded_version:
        start = time.time()
        predictions_flex = flex_model.predict(np.expand_dims(img_np,0))    
        elapsed = time.time()-start
        del img_np
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
                x0 = int(round(float(x-100)/4)+15)
                y0 = int(round(float(y-100)/4)+15)
                probs_flex = np.squeeze(predictions_flex[0, y0, x0, :])
                annotations.append((probs_flex, x, y))
        annotate_and_save(annotations, d, image_file)
        classes_image = annotate_and_save_per_class(annotations, d, image_file)


    labels = load_labels(image_file)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for (c_name, x, y) in labels:
        if 100 <= x < sx-101 and 100 <= y < sy-101:
            x0 = int(round(float(x-100)/4)+15)
            y0 = int(round(float(y-100)/4)+15)
            probs_flex = np.squeeze(predictions_flex[0,y0,x0,:])

            predicted_class = np.argmax(probs_flex)
            c = train_model_mp.get_classes().index(c_name)
            confusion_matrix[c, predicted_class] += 1
    if save_annotations:
        return confusion_matrix, classes_image
    else:
        return confusion_matrix


if __name__ == "__main__":

    np.random.seed(1)
    num_classes = 6
    model_path = "./Results/Saved_Model_As_h5"
    img_path = "./ImageTestSet/201x201_TestSetMidFull"
    setup = utils.generate_setup_dict()

    # each model will have its own folder
    model_names = ['resnet_manual_highres_center_only_f1_{}_f2_{}'.format(int(x), int(2*x)) for x in [2,4,6,8,16]],
    images_dir = './ImageTestSet/201x201_TestSetMidFull/'
    for model_name in model_names:
        try:
            model = keras.models.load_model(model_path + "/" + model_name +".h5")
            flex_model = transform_highres_center_model(model,
                                                          num_classes, 
                                                          model_name=model_name, 
                                                          from_model_to_setup=setup)
            
            sum_of_confusion_matrices = np.zeros((6, 6))
            for image_name in [img for img in os.listdir(images_dir) if img.endswith('.JPG')]:
                confusion_matrix = predict_image_highres(model, flex_model, images_dir + image_name)
                sum_of_confusion_matrices += confusion_matrix
                
            print(sum_of_confusion_matrices)
            np.save('./Results/tmp/sum_of_cm_' + 'flex_' + model_name + '.npy', sum_of_confusion_matrices)
            tf.reset_default_graph()
        except Exception as e:
            print(model_name, e)

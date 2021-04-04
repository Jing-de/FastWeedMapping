import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import time
import train_model as train_model
import evaluate_model as evaluate_model
import glob    
from predict_on_full_images import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def trt_frozen_graph_and_tensors_fp32(model_name, frozen_graph_filepath='./Data/Frozen_graphs/resnet_manual_highres_center_only_f1_2_f2_4_frozen_graph.pb'):
    print('OPENING FROZEN GRAPH FOR MODEL {}.'.format(model_name))
    with open(frozen_graph_filepath, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(frozen_graph)

        input_node = 'import/input1_1'
        output_node = 'import/local_dense/truediv'

        frozen_graph = sess.graph
        x = frozen_graph.get_tensor_by_name(input_node + ':0')
        y = frozen_graph.get_tensor_by_name(output_node + ':0')
        
        return frozen_graph, x, y

    
def check_predictions(image_fp, trt_savedmodel_folder, resnet_file):
    # verify if inferences folder already exist with both files on it
    import os.path
    flex_model_inference_exist = os.path.exists('./Inferences/flex_model_predictions.npy')
    trt_fp16_model_inference_exist = os.path.exists('./Inferences/trt_fp16_model_predictions.npy')
    trt_int8_model_inference_exist = os.path.exists('./Inferences/trt_int8_model_predictions.npy')

    if flex_model_inference_exist == True and trt_fp16_model_inference_exist == True:
        flex_model_pred = np.load('./Inferences/flex_model_predictions.npy')
        trt_fp16_model_pred = np.load('./Inferences/trt_fp16_model_predictions.npy')
        return np.allclose(flex_model_pred, trt_fp16_model_pred, atol=0.025)
    else:
        # run the benchmarks with num_of_runs = 1 and save_inference = True
        return 'TO DO'


def benchmark_trt_model_fp16(image_fp='./Data/201x201_TestSetMidFull/DSC03534.JPG', 
                             trt_savedmodel_folder='./Models/trt_model_fp16_savedmodel/', 
                             num_of_runs=10, 
                             save_inference=False):
    
    """ Gives the time for classifying an image using the trt optimized model
    
    @Args
    image_fp (str) : Filepath for image to be classified
    trt_savedmodel_folder (str) : Path to folder containing the trt SavedModel
    num_of_runs (int) : Number of inferences to be made
    save_inference (Boolean) : If set to True, then it will save the inferences as a pickle file
                                in a folder named "inferences"
    
    @Return
    trt_elapsed (list) : Time taken for each inference
    
    """
    print('Loading image...')
    img = Image.open(image_fp)
    sx, sy = img.size
    input_data = np.expand_dims(img, 0)
    del img
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
        
    with tf.Session(config=config) as sess:
        print('Loading model...')
        # First load the SavedModel into the session    
        tf.saved_model.loader.load(
            sess, 
            [tf.saved_model.tag_constants.SERVING],
           trt_savedmodel_folder)
        print('Model loaded successfully.')
        trt_elapsed = []
        for i in range(num_of_runs+1):
            start = time.time()
            output = sess.run(['local_dense_4/truediv:0'], feed_dict={'input1_6:0': input_data/255.0})
            if i > 1:
                print('Running inference {}'.format(i))
                trt_elapsed.append(time.time() - start)
    del input_data
    
    np.save('./inferences/trt_fp16_elapsed.npy', trt_elapsed)
    if save_inference:
        try:    
            print('Saving trt FP16 predictions...')
            np.save('./Inferences/trt_fp16_model_predictions', output[0])
            print('Saved correctly.')
        except:
            print('Could not save properly.')
            print(type(output), type(output[0]))
    
    return trt_elapsed
    
    
    
def benchmark_flex_model(image_fp='./Data/201x201_TestSetMidFull/DSC03534.JPG', 
                         resnet_file='resnet_highres_center_only.h5', 
                         num_of_runs=10, 
                         save_inference=False):
    
    """ Gives the time for classifying the image using the flex model
    
    @Args
    image_fp (str) : Filepath for image to be classified
    resnet_file (str) : Path to the resnet h5 model
    num_of_runs (int) : Number of inferences to be made
    save_inference (Boolean) : If set to True, then it will save the inferences as a pickle file
                                in a folder named "inferences"
    
    @Return
    flex_elapsed (list) : Time taken for each inference
    
    """
#     from predict_on_full_images import transform_highres_center_model
    import keras
    
    print('Loading image...')
    img = Image.open(image_fp)
    input_data = np.expand_dims(img, 0)
    del img

    flex_model = transform_highres_center_model(keras.models.load_model(resnet_file))
    flex_elapsed = []
    for i in range(num_of_runs+1):
        start = time.time()
        flex_pred = flex_model.predict(input_data/255.0)
        if i > 1:
            flex_elapsed.append(time.time() - start)

    del input_data

    np.save('./Inferences/flex_model_elapsed.npy', flex_elapsed)
    print('Saving flex model predictions...')
    if save_inference:
        try:
            np.save('./Inferences/flex_model_predictions', flex_pred)
            print('Saved correctly.')
        except:
            print('Could not save properly.')
            
    return flex_elapsed



def benchmark_trt_model_int8(image_fp='./Data/201x201_TestSetMidFull/DSC03534.JPG', 
                             flex_savedmodel_fp='./Models/flex_model', 
                             num_of_runs=10, 
                             save_inference=False, 
                             save_annotations=False):
    
    '''
    DISCLAIMER: 
        This model consumes the flex model 'SavedModel', and transforms it to
        INT8 on the fly. 
        It is still a 'to-do' to save the INT8 as SavedModel.
    '''
    
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    from PIL import Image
    import numpy as np
    
    img = Image.open(image_fp)
    input_data = np.expand_dims(img, 0)
    del img
    
    converter = trt.TrtGraphConverter(
        input_saved_model_dir = flex_savedmodel_fp,
        precision_mode=trt.TrtPrecisionMode.INT8
    )
    converter.convert()
    
    # Run calibration n times.
    converted_graph_def = converter.calibrate(
    fetch_names=['local_dense_4/truediv:0'],
    num_runs=1,
    feed_dict_fn=lambda: {'input1_6:0': input_data/255.0})
    
    for n in converted_graph_def.node:
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
            with tf.gfile.GFile("%s.calib_table" % (n.name.replace("/", "_")), 'wb') as f:
                f.write(n.attr["calibration_data"].s)

    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(converted_graph_def)
        
    y = graph.get_tensor_by_name("import/local_dense_4/truediv:0")
    x = graph.get_tensor_by_name("import/input1_6:0")
    
    sess = tf.Session(graph=graph)
    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    import time

    elapsed = []
    for i in range(num_of_runs+1):
        start = time.time()
        result = sess.run(y, feed_dict={x: input_data/255.0})
        if i > 1:
            elapsed.append(time.time() - start)
    
    del input_data
 
    np.save('./Inferences/trt_int8_elapsed.npy', elapsed)
    if save_inference:
        np.save('./Inferences/trt_int8_model_predictions.npy', result)
    
    return elapsed



def annotations_from_inference(inference_results_fp='./Inferences/trt_model_predictions.npy', 
                               image_fp='./Data/201x201_TestSetMidFull/DSC03534.JPG', 
                               save_annotations=False):
    
    """ Takes a (numpy) array and creates annotations from it.
    
    @Args
    inference_results_fp (str) : ./path/to/inference.npy
    image_fp (str) : ./path/to/image.jpg
    
    @Returns
    
    """
    img = Image.open(image_fp)
    sx, sy = img.size
    
    inference_results = np.load(inference_results_fp)

    print("Computing annotations...")
    annotations = []
    d = 4
    for x in range(100,sx-101,d):
        for y in range(100,sy-101,d):
            x0 = int(round(float(x-100)/4)+15)
            y0 = int(round(float(y-100)/4)+15)
            probs_flex = np.squeeze(inference_results[0, y0, x0, :])
            annotations.append((probs_flex, x, y))
    if save_annotations:
        annotate_and_save(annotations, d, image_fp)
        annotate_and_save_per_class(annotations, d, image_fp)
    
    labels = load_labels(image_fp)
    confusion_matrix = np.zeros((6,6))
    for (c_name,x,y) in labels:
        if 100 <= x < sx-101 and 100 <= y < sy-101:
            x0 = int(round(float(x-100)/4)+15)
            y0 = int(round(float(y-100)/4)+15)
            probs_flex = np.squeeze(inference_results[0, y0, x0,:])
            
            predicted_class = np.argmax(probs_flex)
            c = train_model.get_classes().index(c_name)
            confusion_matrix[c, predicted_class] += 1
    evaluate_model.print_statistics(confusion_matrix)
    return confusion_matrix



def generate_benchmark_report(time_elapsed_trt_fp16, 
                              time_elapsed_flex, 
                              time_elapsed_trt_int8,
                              save_report_fp='./Data/benchmark_report/benchmark_02.jpg'):
    plt.figure(figsize=(20,5))
    plt.title('Benchmark - Original model vs Optimized trt models')
    plt.plot([i+1 for i in range(len(time_elapsed_flex))], time_elapsed_flex, '-o', label='flex_model')
    plt.plot([i+1 for i in range(len(time_elapsed_trt_fp16))], time_elapsed_trt_fp16, '-o', label='trt_fp16_model')
    plt.plot([i+1 for i in range(len(time_elapsed_trt_int8))], time_elapsed_trt_int8, '-o', label='trt_int8_model')
    plt.grid()
    plt.xlabel('i-th iteration')
    plt.ylabel('Time taken (seconds)')
    plt.xticks([i+1 for i in range(len(time_elapsed_trt_fp16)+1)])
    plt.legend()
    plt.savefig(save_report_fp)
    return



def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def



def load_image(path='Data/201x201_TestSetMidFull/DSC03534.JPG'):
    ''' Opens the original image, set it to the right dimensions 
        and return as numpy array
    '''
    from PIL import Image
    import numpy as np
    
    img = Image.open(path)
    #new_height = int(img.size[0]*0.8)
    #new_width  = int(new_height * img.size[1] / img.size[0])
    #img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img = np.array(img)/255.
    img_as_np = np.expand_dims(img, 0)

    return img_as_np



def generate_setup_dict():
    """ This utility function configures the tensorflow models, ie, builds all layers
    with the backbone filters used by WeedAI, ie, 2-4, 4-8, and so on
    """
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


def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    """ Takes a graph that was created by Tensorflow and freezes it
    """
    from tensorflow.python.framework import graph_io
    with graph.as_default():
        session.run(tf.global_variables_initializer())
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


def load_graph(frozen_graph_filename):
    """ Loads a saved frozen graph
    """
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='prefix')
    return graph, graph_def


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
    
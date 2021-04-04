# After the generation of FROZEN GRAPHS, we must perform the change in precision mode
# That is, transform models from FP32 to FP16/INT8. 
# Here we load the Frozen Graphs, convert it to FP16
# In this same script, we generate the inferences and save their times to run.
# Further, the evaluation of the inferences can be generated using the "evaluate_model_trt.py"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from utils import load_image
import numpy as np
import time
import traceback
import sys
import os


#TODO: GIVE PARAMETERS FROM THE TERMINAL TO BE RUN
    
TEST_FLAG = True # <- change it to false if you dont want the "TEST" prefix 
SAVE_RESULTS = True
MODEL_NAME_PREFIX = 'resnet_manual_highres_center_only'
FROZEN_GRAPH_FILEPATH = './Models/Frozen_graphs/'

def load_graph_and_convert(model_name, frozen_graph_filepath, precision_mode, seed, f1, f2):
    """ Load a full-precision (FP32) frozen graph, and return it as a half-precision (FP16) frozen graph
    @params:
        model_name (string): The "base" name of your model, such as "resnet", so it can be found in the directory
        frozen_graph_filepath (string): The path to the directory containing your frozen graph
        precision_mode (string): The precision you are converting your model, here we only use 'FP16' so far
    @return:
        The original frozen_graph converted into a different precision mode 
        
    """
    print('OPENING FROZEN GRAPH FOR MODEL {}.'.format(model_name))
 
    frozen_graph_filepath = frozen_graph_filepath + '{}_{}/flex_random_seed_{}_'.format(f1,f2,seed) + model_name + '_frozen_graph.pb'
    with open(frozen_graph_filepath, 'rb') as f:
        frozen_graph_gd = tf.GraphDef()
        frozen_graph_gd.ParseFromString(f.read())

    print('BEGINNING THE CONVERSION TO TRT {}'.format(precision_mode))
    converter = trt.TrtGraphConverter(input_graph_def=frozen_graph_gd,
                                      nodes_blacklist=['local_dense/truediv'], 
                                      precision_mode=precision_mode, 
                                      use_calibration=True,
                                      is_dynamic_op=True)

    try:
        frozen_graph = converter.convert()
        print('CONVERSION FINISHED WITH SUCCESS.')
    except Exception as e:
        errorMessage = 'Exception catched on file: '
        errorMessage += os.path.abspath(sys.argv[0]) + '\n'
        tracebackMessage = traceback.format_exc()
        text_file = open('./Logs/exceptions.txt', 'w')
        text_file.write(errorMessage + tracebackMessage + '\n')
        text_file.close()
        print(errorMessage)
    return frozen_graph

    
def predictions_results_and_time(frozen_graph, config, elapsed_time_dir, predictions_dir, save_results=False):
    """
    Arguments:
        frozen_graph (tf frozen graph): The converted tensorflow FROZEN GRAPH
        config (tf configs): Configurations regarding the Tensorflow-GPU
        elapsed_time_dir (string): Directory to save the elapsed times for inferences
        predictions_dir (string): Directory to save the inferences itself 
                                  The output has shape (1, 797, 1209, 6) for an input of shape (1, 4912, 3264, 3)
                                  Which is the probability vector for each pixel of the output
        save_results (Boolean): Whether to save the output in the directories given above
    """
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        tf.import_graph_def(frozen_graph)

        input_node = 'import/input1_1'
        output_node = 'import/local_dense/truediv'

        frozen_graph = sess.graph
        x = frozen_graph.get_tensor_by_name(input_node + ':0')
        y = frozen_graph.get_tensor_by_name(output_node + ':0')

        del frozen_graph

        print('BEGINNING INFERENCE')
        elapsed_frozen_graph_trt = []
        num_of_runs = 15
        for i in range(num_of_runs + 1):
            start = time.time()
            predictions_flex_frozen_graph = sess.run(y, feed_dict={x:load_image()})
            if i > 0:
                elapsed_frozen_graph_trt.append(time.time() - start)
            if i%10 == 0:
                print('Inference {}'.format(i+1))
        print("Prediction took (on average) %f seconds (inference on full image)" % np.mean(elapsed_frozen_graph_trt))
        
    if save_results:
        np.save(elapsed_time_dir, elapsed_frozen_graph_trt)
        np.save(predictions_dir, predictions_flex_frozen_graph)
    tf.reset_default_graph()
    
def main(save_results, test_flag):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    available_seeds = ['1', '2', '3', '4', '5']
    available_filters = [
        ('2', '4'), ('4','8'), ('6', '12'), ('8', '16'), 
        ('10', '20'), ('12', '24'), ('14','28'), ('16', '32')
    ]

    for seed in available_seeds:
        for (f1, f2) in available_filters:
            model_name = MODEL_NAME_PREFIX + '_f1_{}_f2_{}'.format(f1, f2)
            converted_frozen_graph = load_graph_and_convert(frozen_graph_filepath=FROZEN_GRAPH_FILEPATH, seed=seed, 
                                                            f1=f1, f2=f2, model_name=model_name, precision_mode='FP16')
            preds_dir = './Results/seeds/preds_trt/fp16/{}_{}/flex_'.format(f1, f2) + model_name + '_preds_fp16.npy'
            elaps_dir = './Results/seeds/elapsed_trt/fp16/{}_{}/flex_'.format(f1, f2) + model_name + '_elapsed_fp16.npy'
            if test_flag:
                preds_dir.replace('flex', 'flex_TEST')
                elaps_dir.replace('flex', 'flex_TEST')
            predictions_results_and_time(converted_frozen_graph, config, preds_dir, elaps_dir, save_results)


if __name__ == "__main__":
    main(SAVE_RESULTS, TEST_FLAG)
# This script should be used to generate FROZEN GRAPHS from models saved as format .h5
# The reasoning behind: When training a model, many times we save it as an h5 
# due to being the Keras standard format. However, when using an older version of Tensorflow
# it is better to have a Frozen Graph to perform inference faster. 
# Hence, since there is not a "built-in" way to perform this task, 
# this script provides the conversion.

import sys
import os
import traceback

# global variables
TEST_FLAG = True # <- change this to False if you don't want to generate the prefix "TEST"
PATH_FOR_KERAS_MODELS = './Models/Keras_h5/'
PATH_FOR_FROZEN_GRAPHS_OUTPUTS = './Models/Frozen_graphs/'


def main(path_for_keras_models, path_for_frozen_graph_outputs, seeds, filterCombinations, testFlag=False):
    """
    This script is used for generating models in the old "Frozen Graphs" Tensorflow format, from models
    previously saved as "h5" using Keras. 
    These frozen graphs will be important for generating inferences.
    
    IMPORTANT REMARKS: 
    - We retrieve the "vanilla" trained .h5 and generate a "flex model" frozen graph
    - During the conversion to frozen graph, the frozen graph does not change its precision mode,
      the precision mode conversion happens when calling the `predict_on_full_images.py` 
    
    Arguments:
        path_for_keras_models (str): Path for the directory where you .h5 models are located.
        path_for_frozen_graph_outputs (str): Path for the directory where your frozen graph models will be saved.
        seeds (list of strings): Name of the initialization seed for a trained model, 
                                 eg: resnet_seed_1 is a resnet initialized using np.seed(1) 
        filterCombinations (list of tuples): Resnets has backbones of two kinds of filters, 
                                             eg: ('2','4') is a resnet with 2 filters for convolutions from the layer on group A
                                             and 4 filters for convolutions after layers on group B
        testFlag (Boolean): A flag that adds a "TEST" prefix for the name of the output model for debugging purposes.
    """
    for seed in seeds:
        for (f1, f2) in filterCombinations:
            model_name = 'random_seed_{}_resnet_manual_highres_center_only_f1_{}_f2_{}'.format(seed, f1, f2)
            try:
                input_model_path = path_for_keras_models + 'seeds/{}_{}/'.format(f1, f2, f1, f2) + model_name + '.h5'
                if testFlag:
                    model_name = 'TEST_' + model_name
                output_model_path = path_for_frozen_graph_outputs + '{}_{}/flex_{}_frozen_graph.pb'.format(f1, f2, model_name)
                os.system('python3 keras_to_tensorflow.py --input_model={} --output_model={}'.format(input_model_path, output_model_path))
            except Exception as e:
                errorMessage = 'Exception catched on file: '
                errorMessage += os.path.abspath(sys.argv[0]) + '\n'
                tracebackMessage = traceback.format_exc()
                text_file = open('./Logs/exceptions.txt', 'w')
                text_file.write(errorMessage + tracebackMessage + '\n')
                text_file.close()
                print(errorMessage)
                
            
if __name__ == "__main__":
    available_seeds = ['1', '2', '3', '4', '5']
    available_filters = [('2', '4'), ('4','8'), ('6', '12'), ('8', '16'), 
                        ('10', '20'), ('12', '24'), ('14','28'), ('16', '32')]
    main(PATH_FOR_KERAS_MODELS, PATH_FOR_FROZEN_GRAPHS_OUTPUTS, available_seeds, available_filters, TEST_FLAG)
# 20/01/2020 - This script is not used anymore 
# Under review!

import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from utils import *
    
    
    
# PARAMETERS SETUP
NUM_OF_RUNS = 15
SAVE_INFERENCE = False
SAVE_ANNOTATIONS = False
CHECK_PREDICTIONS = True
IMAGE_FP = './Data/201x201_TestSetMidFull/DSC03534.JPG'
TRT_SAVEDMODEL_FOLDER_FP16 = './Models/trt_model/'
TRT_FROZEN_GRAPH_FILE_INT8 = './tmp/trt_frozen_graph_int8/trt_graph.pb'
RESNET_FILE = 'resnet_highres_center_only.h5'
FLEX_MODEL_PREDS_NPY = './inferences/flex_model_predictions.npy'
TRT_FP16_MODEL_PREDS_NPY = './inferences/trt_fp16_model_predictions.npy'
TRT_INT8_MODEL_PREDS_NPY = './inferences/trt_int8_model_predictions.npy'



def main():
    trt_fp16_elapsed = benchmark_trt_model_fp16(num_of_runs=NUM_OF_RUNS, save_inference=SAVE_INFERENCE)
    flexmodel_elapsed = benchmark_flex_model(num_of_runs=NUM_OF_RUNS, save_inference=SAVE_INFERENCE)
    trt_int8_elapsed = benchmark_trt_model_int8(num_of_runs=NUM_OF_RUNS, save_inference=SAVE_INFERENCE)

    return trt_fp16_elapsed, flexmodel_elapsed, trt_int8_elapsed



if __name__ == "__main__":
    trt_fp16_elapsed, flexmodel_elapsed, trt_int8_elapsed = main()
    print('The time taken for the original (flex) model to run was:', flexmodel_elapsed, '\n')
    print('The time taken for the trt fp16 optimized to run was:', trt_fp16_elapsed, '\n')
    print('The time taken for the trt fp16 optimized to run was:', trt_int8_elapsed, '\n')

    print('Generating the benchmark graph')
    generate_benchmark_report(trt_fp16_elapsed, 
                              flexmodel_elapsed, 
                              trt_int8_elapsed,
                              save_report_fp='./Data/benchmark_report/benchmark_02.jpg')

    confusion_matrix_flex = annotations_from_inference(inference_results_fp=FLEX_MODEL_PREDS_NPY, save_annotations=False)
#     confusion_matrix_trt_fp16 = annotations_from_inference(inference_results_fp=TRT_FP16_MODEL_PREDS_NPY, save_annotations=False)
    confusion_matrix_trt_int8 = annotations_from_inference(inference_results_fp=TRT_INT8_MODEL_PREDS_NPY, save_annotations=False)
#     confusion_matrix_trt_int8 = annotations_from_inference(inference_results_fp='./inferences/TESTE.npy', save_annotations=True)

    if (np.diag(confusion_matrix_flex) == np.diag(confusion_matrix_trt_int8)).all():
        print('The trace of the confusion matrices for both models are equal')
    else:
        print('The trace of the confusion matrices are not equal, and the difference is {}'\
              .format(abs(np.trace(confusion_matrix_flex) - np.trace(confusion_matrix_trt_int8))))